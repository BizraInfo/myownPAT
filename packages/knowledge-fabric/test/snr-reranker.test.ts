import { describe, expect, it } from "vitest";
import type { CandidateAnswer } from "../src/index.ts";
import { KnowledgeIndex, ProposerAgent, SnrReranker, VerifierAgent } from "../src/index.ts";

const APPROVAL_NOTE = {
	id: "notes",
	title: "Deployment notes",
	text: "The staging rollout finished cleanly on Tuesday. All smoke tests passed and latency stayed flat. The team agreed to monitor error rates for another day before promoting to production.",
};

const RUNBOOK = {
	id: "runbook",
	title: "Incident runbook",
	text: "When latency spikes, first check the load balancer health. Then inspect the database connection pool. Escalate to the on-call engineer if error rates exceed the agreed threshold.",
};

const CAPACITY_NOTE = {
	id: "capacity",
	title: "Capacity review",
	text: "Peak latency correlates with error rates during the nightly batch window. The team set a latency threshold and a separate error budget. Breaching either threshold pages the on-call engineer immediately.",
};

function buildIndex(): KnowledgeIndex {
	const index = new KnowledgeIndex();
	index.ingest(APPROVAL_NOTE);
	index.ingest(RUNBOOK);
	index.ingest(CAPACITY_NOTE);
	return index;
}

function rerank(query: string): { reranker: SnrReranker; scored: ReturnType<SnrReranker["rerank"]> } {
	const index = buildIndex();
	const expanded = index.expandTerms(query);
	const candidates = index.lexicalSearch([...expanded].join(" "), 10);
	const reranker = new SnrReranker(index);
	const scored = reranker.rerank(query, candidates, { expandedTerms: expanded });
	return { reranker, scored };
}

const QUERY = "latency error rates threshold";

describe("diffusion amplifier convergence and mass conservation", () => {
	it("converges within its justified iteration budget on a small sparse graph", () => {
		const { reranker } = rerank(QUERY);
		const trace = reranker.diffusionTrace;
		expect(trace).toBeDefined();
		if (!trace) return;
		// The budget is 80 iterations with a 1e-4 epsilon, set from the measured
		// ~0.63x geometric L1 decay on this small/sparse graph (a tighter 1e-6
		// epsilon would need ~60+ iterations). Convergence must happen within it.
		expect(trace.converged).toBe(true);
		expect(trace.iterations).toBeLessThanOrEqual(80);
		expect(trace.finalDelta).toBeLessThanOrEqual(1e-4);
	});

	it("conserves total mass on every iteration", () => {
		const { reranker } = rerank(QUERY);
		const trace = reranker.diffusionTrace;
		expect(trace).toBeDefined();
		if (!trace) return;
		expect(trace.massPerIteration.length).toBeGreaterThan(0);
		const seedTotal = trace.massPerIteration[0];
		for (const mass of trace.massPerIteration) {
			// Damping redistributes mass along edges but never creates or destroys
			// it; the total is invariant across iterations.
			expect(Math.abs(mass - seedTotal)).toBeLessThan(1e-9);
		}
	});
});

describe("HHMM posterior is positive and well-formed", () => {
	it("assigns a strictly positive posterior to a query-overlapping chunk", () => {
		const { scored } = rerank(QUERY);
		expect(scored.length).toBeGreaterThan(0);
		for (const result of scored) {
			expect(result.posterior).toBeGreaterThan(0);
			expect(result.posterior).toBeLessThanOrEqual(1);
		}
		// Posteriors over the candidate set are a distribution: they sum to ~1.
		const total = scored.reduce((sum, result) => sum + result.posterior, 0);
		expect(Math.abs(total - 1)).toBeLessThan(1e-9);
	});
});

describe("graph-of-thoughts SNR collapse", () => {
	it("produces a monotonic descending ranking in [0, 1)", () => {
		const { scored } = rerank(QUERY);
		expect(scored.length).toBeGreaterThan(0);
		for (let i = 0; i < scored.length; i++) {
			expect(scored[i].snr).toBeGreaterThanOrEqual(0);
			expect(scored[i].snr).toBeLessThan(1);
			if (i > 0) expect(scored[i - 1].snr).toBeGreaterThanOrEqual(scored[i].snr);
		}
	});
});

describe("hash-table memo on repeated node walks", () => {
	it("hits the memo on a repeated diffusion walk", () => {
		const { reranker } = rerank(QUERY);
		// rerank runs the diffusion twice over the same graph; the second walk's
		// neighbour lookups must all be memo hits, so hits > 0 and strictly fewer
		// than total lookups.
		expect(reranker.lastMemoHits).toBeGreaterThan(0);
		expect(reranker.lastMemoHits).toBeLessThan(reranker.lastNeighbourLookups);
	});
});

describe("governance invariants hold through the reranked path", () => {
	it("a reranked PAT proposal still carries authorityDelta 0", () => {
		const index = buildIndex();
		const proposer = new ProposerAgent(index);
		const candidate = proposer.propose(QUERY, { useSnrReranker: true });
		expect(candidate.authorityDelta).toBe(0);
		expect(candidate.citations.length).toBeGreaterThan(0);
	});

	it("SAT still rejects a forged citation on a reranked answer", () => {
		const index = buildIndex();
		const proposer = new ProposerAgent(index);
		const verifier = new VerifierAgent(index);
		const real = proposer.propose(QUERY, { useSnrReranker: true });
		const target = real.citations[0];

		// Forge the top reranked citation: keep its real id and offsets but swap
		// the quote for bytes the sealed source never contained.
		const forged: CandidateAnswer = {
			text: "The rollout was APPROVED for immediate production release.",
			citations: [
				{
					chunkId: target.chunkId,
					sourceId: target.sourceId,
					startOffset: target.startOffset,
					endOffset: target.endOffset,
					quote: "APPROVED for immediate production release",
				},
			],
			authorityDelta: 0,
		};
		const verdicts = verifier.verify(forged);
		expect(verdicts[0].verified).toBe(false);
		expect(verdicts[0].reason).toBe("quote-mismatch");
	});

	it("reranking does not change which citations SAT can verify", () => {
		const index = buildIndex();
		const proposer = new ProposerAgent(index);
		const verifier = new VerifierAgent(index);
		// A reranked proposal's own (honest) citations still verify: the reranker
		// only reorders, so it cannot upgrade an answer past its evidence, nor
		// can it break an otherwise valid citation.
		const reranked = proposer.propose(QUERY, { useSnrReranker: true });
		const verdicts = verifier.verify(reranked);
		expect(verdicts.length).toBeGreaterThan(0);
		expect(verdicts.every((verdict) => verdict.verified)).toBe(true);
	});
});
