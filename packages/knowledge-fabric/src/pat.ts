import type { ChunkScore, KnowledgeIndex } from "./index-store.ts";
import { SnrReranker } from "./snr-reranker.ts";
import type { CandidateAnswer, Citation } from "./types.ts";

/**
 * PAT: the proposer. Does the retrieval (lexical ranking plus a one-hop graph
 * walk over the candidate graph) and composes a candidate cited answer.
 *
 * PAT has no authority. Every candidate it returns carries
 * `authorityDelta === 0`, and that field is literally typed to `0`, so PAT
 * cannot emit a candidate that grants authority. PAT also never verifies its
 * own citations; it merely claims them. SAT proves or rejects them.
 */

export interface ProposeOptions {
	/** Max chunks to draw into the candidate answer. Defaults to 3. */
	readonly maxChunks?: number;
	/** Whether to expand the query via the candidate graph. Defaults to true. */
	readonly useGraphWalk?: boolean;
	/**
	 * Whether to reorder the lexical candidates with the HHMM + diffusion SNR
	 * reranker before composing citations. Opt-in and defaults to false so the
	 * baseline TF-IDF behaviour is unchanged. The reranker only reorders; it
	 * mints no authority and cannot strengthen an answer past SAT, which still
	 * re-reads the sealed bytes for every citation.
	 */
	readonly useSnrReranker?: boolean;
}

const DEFAULT_MAX_CHUNKS = 3;

export class ProposerAgent {
	private readonly index: KnowledgeIndex;
	private readonly reranker: SnrReranker;

	constructor(index: KnowledgeIndex) {
		this.index = index;
		this.reranker = new SnrReranker(index);
	}

	propose(query: string, options: ProposeOptions = {}): CandidateAnswer {
		const maxChunks = options.maxChunks ?? DEFAULT_MAX_CHUNKS;
		const useGraphWalk = options.useGraphWalk ?? true;
		const useSnrReranker = options.useSnrReranker ?? false;

		const expandedTerms = useGraphWalk ? this.index.expandTerms(query) : undefined;
		const effectiveQuery = expandedTerms ? [...expandedTerms].join(" ") : query;
		let ranked: readonly ChunkScore[] = this.index.lexicalSearch(effectiveQuery, maxChunks);

		if (useSnrReranker && ranked.length > 0) {
			// Reorder by SNR but keep the same candidate set and ChunkScore shape so
			// the downstream citation composition is identical except for order.
			const byId = new Map(ranked.map((result) => [result.chunk.id, result]));
			const reranked = this.reranker.rerank(query, ranked, { expandedTerms });
			ranked = reranked.map((result) => byId.get(result.chunk.id)).filter((result) => result !== undefined);
		}

		const citations: Citation[] = ranked.map((result) => ({
			chunkId: result.chunk.id,
			sourceId: result.chunk.sourceId,
			startOffset: result.chunk.startOffset,
			endOffset: result.chunk.endOffset,
			quote: result.chunk.text,
		}));

		const text =
			citations.length === 0
				? ""
				: ranked.map((result, position) => `[${position + 1}] ${result.chunk.text.trim()}`).join("\n");

		return { text, citations, authorityDelta: 0 };
	}
}
