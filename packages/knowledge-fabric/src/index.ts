/**
 * @earendil-works/pi-knowledge-fabric
 *
 * A self-contained, framework-agnostic knowledge fabric built around a
 * governance split rather than a single retrieval algorithm:
 *
 *   ingest -> local lexical index -> PAT propose -> SAT verify -> DEMA label
 *
 * - DEMA is the query-in / labelled-answer-out surface.
 * - PAT proposes a cited answer and may only propose (authorityDelta === 0).
 * - SAT independently verifies every citation against the sealed source bytes,
 *   model-blind, and rejects the answer if any citation is forged or unprovable.
 *
 * Invariants: no retrieved chunk may authorize an action, and an answer is
 * never stronger than its weakest cited evidence.
 *
 * This package depends on no other workspace package and is not wired into any
 * runtime. It exposes a thin public API: the ingest/query surface plus the
 * agents and their types.
 */

export type { KnowledgeFabricOptions } from "./dema.ts";
export { KnowledgeFabric } from "./dema.ts";
export type { ChunkScore, IngestOptions } from "./index-store.ts";
export { KnowledgeIndex } from "./index-store.ts";
export type { ProposeOptions } from "./pat.ts";
export { ProposerAgent } from "./pat.ts";
export { VerifierAgent } from "./sat.ts";
export type { DiffusionTrace, SnrRerankerOptions, SnrScore } from "./snr-reranker.ts";
export { SnrReranker } from "./snr-reranker.ts";
export type {
	CandidateAnswer,
	Citation,
	CitationRejectionReason,
	CitationVerdict,
	LabelledResult,
	SealedChunk,
	SourceDocument,
	TruthLabel,
} from "./types.ts";
export { sourceDocumentSchema } from "./types.ts";
