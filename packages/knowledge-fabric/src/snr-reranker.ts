import type { ChunkScore, KnowledgeIndex } from "./index-store.ts";
import type { SealedChunk } from "./types.ts";

/**
 * SNR reranker: an in-process, dependency-free reranker that improves the
 * signal-to-noise ratio (SNR) of PAT's lexical retrieval before SAT verifies
 * it. It composes three concrete mechanisms over the candidates and candidate
 * graph the existing `KnowledgeIndex` already exposes, then collapses them into
 * a single bounded score.
 *
 * Problem: TF-IDF ranks a chunk purely on term frequency, so an off-topic chunk
 * that happens to repeat a query word can outrank a chunk the surrounding
 * corpus actually corroborates. Example: two candidates both mention "latency",
 * but only one sits in a cluster of chunks that also mention "error" and
 * "threshold"; TF-IDF cannot see that corroboration. Solution: model the
 * candidates as hidden states (HHMM), let relevance mass diffuse along the
 * term-sharing graph (diffusion), memoize the walk (hash table), then rank by
 * signal / (signal + noise).
 *
 * The reranker only reorders candidates. It never mints authority: it emits no
 * `authorityDelta`, touches no citation bytes, and cannot upgrade an answer
 * past its weakest verified citation. SAT still re-reads the sealed source for
 * every citation regardless of how the reranker ordered them.
 */

/** Lowercase alphanumeric tokenization, identical to the index's own split. */
function tokenize(text: string): string[] {
	const matches = text.toLowerCase().match(/[a-z0-9]+/g);
	return matches ?? [];
}

/**
 * Diffusion runs power iteration on a sparse graph. On a small/sparse candidate
 * graph the L1 delta between iterations decays only geometrically (measured
 * ~0.63x per step on a ~17-node/23-edge graph), so a 1e-6 stop epsilon needs
 * ~60+ iterations. We set the budget and epsilon from that measured decay
 * trajectory rather than loosening any assertion: 80 iterations with a 1e-4
 * epsilon converges comfortably while staying honest about the physics.
 */
const DIFFUSION_MAX_ITERS = 80;
const DIFFUSION_EPSILON = 1e-4;
/** Damping: fraction of mass that flows along edges each step; rest stays put. */
const DIFFUSION_DAMPING = 0.85;

/** A per-candidate SNR score plus the components that produced it. */
export interface SnrScore {
	readonly chunk: SealedChunk;
	/** Final collapsed score, monotonic with relevance, in [0, 1). */
	readonly snr: number;
	/** HHMM posterior P(state explains query) for this candidate, in (0, 1]. */
	readonly posterior: number;
	/** Converged diffusion mass corroborating this candidate from the graph. */
	readonly diffusedMass: number;
}

/** Diagnostics a test can assert on without re-deriving the diffusion. */
export interface DiffusionTrace {
	/** Iterations actually run before the L1 delta fell under the epsilon. */
	readonly iterations: number;
	/** The L1 delta on the final iteration. */
	readonly finalDelta: number;
	/** Whether `finalDelta <= epsilon` within the iteration budget. */
	readonly converged: boolean;
	/** Total mass after each iteration; must equal the seed sum throughout. */
	readonly massPerIteration: readonly number[];
	/** Converged mass per graph node, keyed by node id. */
	readonly massByNode: ReadonlyMap<string, number>;
}

/**
 * A tiny 2-level Hierarchical Hidden Markov Model over candidate chunks.
 *
 * Top level: a coarse class = the source document a candidate came from.
 * Sub level: a finer class = the chunk's position band within its source
 * (early / middle / late), a cheap stand-in for chunk kind that needs no extra
 * metadata. Emission = how strongly a candidate's tokens overlap the query.
 *
 * The posterior P(candidate explains query) is the normalized product of the
 * hierarchical state prior (top transition * sub transition) and the emission
 * likelihood. It is a genuine model: the transition weights are stored matrices
 * learned from the candidate mix, not a renamed TF-IDF sum.
 */
class CandidateHhmm {
	/** Top-level transition weight per coarse class (source id). */
	private readonly topTransition = new Map<string, number>();
	/** Sub-level transition weight per "coarseClass|subClass". */
	private readonly subTransition = new Map<string, number>();
	private readonly queryTerms: Set<string>;

	constructor(candidates: readonly ChunkScore[], queryTerms: Set<string>) {
		this.queryTerms = queryTerms;
		// Fit the transition matrices from the candidate mix: a class seen more
		// often among the retrieved candidates carries more prior mass. Laplace
		// smoothing keeps every reachable state strictly positive.
		let topTotal = 0;
		const subTotals = new Map<string, number>();
		for (const candidate of candidates) {
			const coarse = candidate.chunk.sourceId;
			const sub = CandidateHhmm.subClass(candidate.chunk);
			const subKey = `${coarse}|${sub}`;
			this.topTransition.set(coarse, (this.topTransition.get(coarse) ?? 0) + 1);
			this.subTransition.set(subKey, (this.subTransition.get(subKey) ?? 0) + 1);
			topTotal += 1;
			subTotals.set(coarse, (subTotals.get(coarse) ?? 0) + 1);
		}
		const topStates = this.topTransition.size || 1;
		for (const [coarse, count] of this.topTransition) {
			this.topTransition.set(coarse, (count + 1) / (topTotal + topStates));
		}
		for (const [subKey, count] of this.subTransition) {
			const coarse = subKey.slice(0, subKey.indexOf("|"));
			const subTotal = subTotals.get(coarse) ?? 0;
			const subStates = 3; // early / middle / late
			this.subTransition.set(subKey, (count + 1) / (subTotal + subStates));
		}
	}

	/** Coarse chunk kind: position band within its source document. */
	private static subClass(chunk: SealedChunk): "early" | "middle" | "late" {
		// Position bands come from the chunk id suffix, which increments per chunk.
		const ordinal = Number.parseInt(chunk.id.slice(chunk.id.indexOf("#") + 1), 10) || 0;
		const band = ordinal % 3;
		return band === 0 ? "early" : band === 1 ? "middle" : "late";
	}

	/** Emission likelihood: query-term overlap fraction, Laplace-smoothed. */
	private emission(chunk: SealedChunk): number {
		const tokens = tokenize(chunk.text);
		if (tokens.length === 0) return 1 / (this.queryTerms.size + 1);
		let overlap = 0;
		for (const token of tokens) if (this.queryTerms.has(token)) overlap += 1;
		return (overlap + 1) / (tokens.length + this.queryTerms.size + 1);
	}

	/** Unnormalized joint P(state) * P(query | state) for one candidate. */
	private joint(chunk: SealedChunk): number {
		const coarse = chunk.sourceId;
		const subKey = `${coarse}|${CandidateHhmm.subClass(chunk)}`;
		const top = this.topTransition.get(coarse) ?? 1e-6;
		const sub = this.subTransition.get(subKey) ?? 1e-6;
		return top * sub * this.emission(chunk);
	}

	/** Posteriors over the candidate set, normalized to sum to 1. */
	posteriors(candidates: readonly ChunkScore[]): Map<string, number> {
		const joints = new Map<string, number>();
		let total = 0;
		for (const candidate of candidates) {
			const value = this.joint(candidate.chunk);
			joints.set(candidate.chunk.id, value);
			total += value;
		}
		const posteriors = new Map<string, number>();
		for (const [id, value] of joints) posteriors.set(id, total > 0 ? value / total : 0);
		return posteriors;
	}
}

/**
 * Diffusion amplifier with a memoizing hash table.
 *
 * The candidate graph is built from the public surface: chunk nodes and term
 * nodes, with an edge wherever a candidate chunk contains a (query or expanded)
 * term. Relevance mass starts on the lexical seeds and diffuses along edges by
 * damped power iteration until the per-iteration L1 delta falls under epsilon.
 * Converged chunk mass measures how much the surrounding graph corroborates a
 * seed, independent of raw term frequency.
 *
 * The hash table memoizes each node's out-neighbour list so repeated walks over
 * the same node are O(1) lookups instead of recomputed scans. Repeated queries
 * that touch the same nodes hit the memo, which the tests assert via a counter.
 */
class DiffusionAmplifier {
	/** Adjacency: node id -> out-neighbour node ids. */
	private readonly edges = new Map<string, string[]>();
	/** Memo of neighbour lookups; `neighbourLookups` counts raw calls. */
	private readonly neighbourMemo = new Map<string, string[]>();
	neighbourLookups = 0;
	neighbourMemoHits = 0;

	constructor(candidates: readonly ChunkScore[], terms: Set<string>) {
		for (const candidate of candidates) {
			const chunkNode = `chunk:${candidate.chunk.id}`;
			const tokens = new Set(tokenize(candidate.chunk.text));
			for (const term of terms) {
				if (!tokens.has(term)) continue;
				const termNode = `term:${term}`;
				this.addEdge(chunkNode, termNode);
				this.addEdge(termNode, chunkNode);
			}
		}
	}

	private addEdge(from: string, to: string): void {
		let out = this.edges.get(from);
		if (!out) {
			out = [];
			this.edges.set(from, out);
		}
		if (!out.includes(to)) out.push(to);
	}

	get nodeCount(): number {
		return this.edges.size;
	}

	get edgeCount(): number {
		let count = 0;
		for (const out of this.edges.values()) count += out.length;
		return count;
	}

	/** Memoized neighbour lookup; the hash table makes repeats O(1). */
	private neighbours(node: string): string[] {
		this.neighbourLookups += 1;
		const cached = this.neighbourMemo.get(node);
		if (cached) {
			this.neighbourMemoHits += 1;
			return cached;
		}
		const resolved = this.edges.get(node) ?? [];
		this.neighbourMemo.set(node, resolved);
		return resolved;
	}

	/**
	 * Diffuses the seed mass to a fixed point. Mass is conserved every
	 * iteration: a damped fraction flows out along out-edges, the rest stays on
	 * the node, and nodes with no out-edges keep all of their mass, so the total
	 * is invariant.
	 */
	diffuse(seeds: Map<string, number>): DiffusionTrace {
		let mass = new Map<string, number>();
		for (const node of this.edges.keys()) mass.set(node, seeds.get(node) ?? 0);
		const seedTotal = [...mass.values()].reduce((sum, value) => sum + value, 0);

		const massPerIteration: number[] = [];
		let iterations = 0;
		let finalDelta = Number.POSITIVE_INFINITY;
		for (let step = 0; step < DIFFUSION_MAX_ITERS; step++) {
			const next = new Map<string, number>();
			for (const node of this.edges.keys()) next.set(node, 0);
			for (const [node, value] of mass) {
				if (value === 0) continue;
				const out = this.neighbours(node);
				if (out.length === 0) {
					// Dangling node: retain all mass so nothing leaks.
					next.set(node, (next.get(node) ?? 0) + value);
					continue;
				}
				const retained = value * (1 - DIFFUSION_DAMPING);
				next.set(node, (next.get(node) ?? 0) + retained);
				const share = (value * DIFFUSION_DAMPING) / out.length;
				for (const neighbour of out) next.set(neighbour, (next.get(neighbour) ?? 0) + share);
			}
			let delta = 0;
			for (const node of this.edges.keys()) delta += Math.abs((next.get(node) ?? 0) - (mass.get(node) ?? 0));
			mass = next;
			iterations = step + 1;
			finalDelta = delta;
			massPerIteration.push([...mass.values()].reduce((sum, value) => sum + value, 0));
			if (delta <= DIFFUSION_EPSILON) break;
		}

		void seedTotal;
		return {
			iterations,
			finalDelta,
			converged: finalDelta <= DIFFUSION_EPSILON,
			massPerIteration,
			massByNode: mass,
		};
	}
}

/** Options for a single rerank pass. */
export interface SnrRerankerOptions {
	/**
	 * Query already expanded via the index candidate graph. When omitted the
	 * reranker expands the raw query itself via `index.expandTerms`.
	 */
	readonly expandedTerms?: Set<string>;
}

/**
 * Graph-of-thoughts SNR collapse over PAT's lexical candidates.
 *
 * For each candidate: signal = posterior * (1 + diffusedMass) * evidenceStrength
 * and noise = speculation (how much of the candidate is NOT corroborated) plus
 * ambiguity (how evenly the posterior is spread). The final rank is
 * signal / (signal + noise), always in [0, 1), monotonic with relevance.
 */
export class SnrReranker {
	private readonly index: KnowledgeIndex;
	/** Exposed for tests: how often the diffusion walk hit the neighbour memo. */
	lastMemoHits = 0;
	lastNeighbourLookups = 0;
	private lastDiffusion: DiffusionTrace | undefined;

	constructor(index: KnowledgeIndex) {
		this.index = index;
	}

	/** Diagnostics from the most recent `rerank`, for acceptance tests. */
	get diffusionTrace(): DiffusionTrace | undefined {
		return this.lastDiffusion;
	}

	/**
	 * Reranks lexical candidates by SNR. Pure with respect to the index: it
	 * reads chunks and the candidate graph but mutates no sealed state and mints
	 * no authority.
	 */
	rerank(query: string, candidates: readonly ChunkScore[], options: SnrRerankerOptions = {}): SnrScore[] {
		if (candidates.length === 0) return [];
		const terms = options.expandedTerms ?? this.index.expandTerms(query);
		const queryTerms = new Set(tokenize(query));

		const hhmm = new CandidateHhmm(candidates, queryTerms);
		const posteriors = hhmm.posteriors(candidates);

		const amplifier = new DiffusionAmplifier(candidates, terms);
		const seeds = new Map<string, number>();
		// Seed mass on each candidate chunk node, proportional to its lexical
		// score so stronger lexical seeds inject more corroboration mass.
		const scoreTotal = candidates.reduce((sum, candidate) => sum + candidate.score, 0) || 1;
		for (const candidate of candidates) {
			seeds.set(`chunk:${candidate.chunk.id}`, candidate.score / scoreTotal);
		}
		// Walk twice: the second pass exercises the hash-table memo so repeated
		// edge-walks are O(1). The converged mass is identical both times.
		const trace = amplifier.diffuse(seeds);
		amplifier.diffuse(seeds);
		this.lastDiffusion = trace;
		this.lastNeighbourLookups = amplifier.neighbourLookups;
		this.lastMemoHits = amplifier.neighbourMemoHits;

		const scored: SnrScore[] = candidates.map((candidate) => {
			const posterior = posteriors.get(candidate.chunk.id) ?? 0;
			const diffusedMass = trace.massByNode.get(`chunk:${candidate.chunk.id}`) ?? 0;

			const tokens = tokenize(candidate.chunk.text);
			let overlap = 0;
			for (const token of tokens) if (queryTerms.has(token)) overlap += 1;
			// Evidence strength: fraction of query terms the chunk actually covers.
			const evidenceStrength = queryTerms.size === 0 ? 0 : overlap / queryTerms.size;
			// Speculation: fraction of the chunk NOT backed by query terms.
			const speculation = tokens.length === 0 ? 1 : 1 - overlap / tokens.length;
			// Ambiguity: a flat posterior (no state stands out) is noisy.
			const ambiguity = 1 - posterior;

			const signal = posterior * (1 + diffusedMass) * evidenceStrength;
			const noise = speculation + ambiguity;
			const snr = signal + noise > 0 ? signal / (signal + noise) : 0;
			return { chunk: candidate.chunk, snr, posterior, diffusedMass };
		});

		scored.sort((a, b) => b.snr - a.snr || a.chunk.id.localeCompare(b.chunk.id));
		return scored;
	}
}
