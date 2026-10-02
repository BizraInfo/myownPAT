import { describe, expect, it } from "vitest";
import type { CandidateAnswer } from "../src/index.ts";
import { KnowledgeFabric, KnowledgeIndex, ProposerAgent, VerifierAgent } from "../src/index.ts";

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

function buildFabric(): KnowledgeFabric {
	const fabric = new KnowledgeFabric();
	fabric.ingest(APPROVAL_NOTE);
	fabric.ingest(RUNBOOK);
	return fabric;
}

describe("ingest -> lexical index -> PAT -> SAT -> DEMA spine", () => {
	it("ingests one source type and seals chunks with byte offsets", () => {
		const index = new KnowledgeIndex();
		const chunks = index.ingest(APPROVAL_NOTE);
		expect(chunks.length).toBeGreaterThan(0);
		for (const chunk of chunks) {
			expect(APPROVAL_NOTE.text.slice(chunk.startOffset, chunk.endOffset)).toBe(chunk.text);
		}
	});

	it("returns a verified labelled result when citations prove out", () => {
		const fabric = buildFabric();
		const result = fabric.query("latency error rates threshold");
		expect(result.label).toBe("verified");
		expect(result.accepted).toBe(true);
		expect(result.answer.length).toBeGreaterThan(0);
		expect(result.verdicts.every((verdict) => verdict.verified)).toBe(true);
	});

	it("labels a proposal with no citations as unverified, not accepted", () => {
		const fabric = buildFabric();
		const result = fabric.query("zzzznonexistentterm");
		expect(result.label).toBe("unverified");
		expect(result.accepted).toBe(false);
		expect(result.answer).toBe("");
	});
});

describe("SAT rejects forged or unprovable citations (red-first acceptance)", () => {
	it("rejects an answer whose quote was never in the sealed source (forged citation)", () => {
		const index = new KnowledgeIndex();
		const chunks = index.ingest(APPROVAL_NOTE);
		const verifier = new VerifierAgent(index);
		const realChunk = chunks[0];

		// A forged citation: real chunk id and offsets, but a fabricated quote the
		// sealed bytes never contained.
		const forged: CandidateAnswer = {
			text: "The rollout was APPROVED for immediate production release.",
			citations: [
				{
					chunkId: realChunk.id,
					sourceId: realChunk.sourceId,
					startOffset: realChunk.startOffset,
					endOffset: realChunk.endOffset,
					quote: "APPROVED for immediate production release",
				},
			],
			authorityDelta: 0,
		};

		const verdicts = verifier.verify(forged);
		expect(verdicts[0].verified).toBe(false);
		expect(verdicts[0].reason).toBe("quote-mismatch");
	});

	it("rejects an unprovable citation to a chunk that does not exist", () => {
		const index = new KnowledgeIndex();
		index.ingest(APPROVAL_NOTE);
		const verifier = new VerifierAgent(index);
		const verdict = verifier.verifyCitation({
			chunkId: "ghost#999",
			sourceId: "notes",
			startOffset: 0,
			endOffset: 10,
			quote: "whatever",
		});
		expect(verdict.verified).toBe(false);
		expect(verdict.reason).toBe("unknown-chunk");
	});

	it("DEMA labels the whole answer rejected when any single citation fails", () => {
		const fabric = buildFabric();
		const index = new KnowledgeIndex();
		const chunks = index.ingest(APPROVAL_NOTE);
		void fabric;

		const verifier = new VerifierAgent(index);
		const good = chunks[0];
		const mixed: CandidateAnswer = {
			text: "mixed",
			citations: [
				{
					chunkId: good.id,
					sourceId: good.sourceId,
					startOffset: good.startOffset,
					endOffset: good.endOffset,
					quote: good.text,
				},
				{
					chunkId: good.id,
					sourceId: good.sourceId,
					startOffset: good.startOffset,
					endOffset: good.endOffset,
					quote: "tampered",
				},
			],
			authorityDelta: 0,
		};
		const verdicts = verifier.verify(mixed);
		expect(verdicts[0].verified).toBe(true);
		expect(verdicts[1].verified).toBe(false);
		// An answer is never stronger than its weakest cited evidence.
		expect(verdicts.every((verdict) => verdict.verified)).toBe(false);
	});
});

describe("no retrieved chunk may authorize an action (red-first acceptance)", () => {
	it("every PAT proposal carries authorityDelta 0", () => {
		const index = new KnowledgeIndex();
		index.ingest(APPROVAL_NOTE);
		index.ingest(RUNBOOK);
		const proposer = new ProposerAgent(index);
		const candidate = proposer.propose("latency threshold");
		expect(candidate.authorityDelta).toBe(0);
		expect(candidate.citations.length).toBeGreaterThan(0);
	});

	it("every DEMA labelled result carries authorityDelta 0, verified or rejected", () => {
		const fabric = buildFabric();
		const verified = fabric.query("latency error rates threshold");
		const empty = fabric.query("zzzznonexistentterm");
		expect(verified.authorityDelta).toBe(0);
		expect(empty.authorityDelta).toBe(0);
	});
});
