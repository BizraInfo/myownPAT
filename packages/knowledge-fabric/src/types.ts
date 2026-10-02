import { type Static, Type } from "typebox";

/**
 * Shared domain types and typebox schemas for the knowledge fabric.
 *
 * The fabric enforces a governance split between proposing an answer and
 * accepting it. These types name the roles and the invariants that keep a
 * proposer from granting itself authority:
 *
 * - PAT (proposer) retrieves and drafts a cited answer. It may only propose,
 *   so every candidate it emits carries `authorityDelta === 0`.
 * - SAT (verifier) re-checks each citation against the sealed source bytes,
 *   model-blind, and rejects the answer if any citation is forged or
 *   unprovable.
 * - DEMA (surface) takes intent in and returns a labelled result out.
 *
 * One standing invariant: no retrieved chunk may ever authorize an action.
 * A chunk is evidence, never an authority grant. The `authorityDelta` field is
 * literally typed to `0` so the type system forbids a nonzero grant.
 */

/** The only source type the first slice ingests: an in-memory text document. */
export const sourceDocumentSchema = Type.Object(
	{
		/** Stable identifier for the source. Used to seal and later re-read bytes. */
		id: Type.String({ minLength: 1 }),
		/** Raw UTF-8 text of the document. These bytes are sealed verbatim. */
		text: Type.String(),
		/** Optional human-facing title shown by DEMA alongside citations. */
		title: Type.Optional(Type.String()),
	},
	{ additionalProperties: false },
);

export type SourceDocument = Static<typeof sourceDocumentSchema>;

/**
 * A sealed chunk: a contiguous span of a source document together with the
 * exact byte offsets it covers and a content hash of those bytes. SAT verifies
 * citations by re-reading `[startOffset, endOffset)` from the sealed source and
 * comparing the hash, so the chunk can never silently drift from the source.
 */
export interface SealedChunk {
	/** Unique chunk identifier, stable for the lifetime of the index. */
	readonly id: string;
	/** Identifier of the source document this chunk was cut from. */
	readonly sourceId: string;
	/** Byte offset (inclusive) where the chunk begins in the source text. */
	readonly startOffset: number;
	/** Byte offset (exclusive) where the chunk ends in the source text. */
	readonly endOffset: number;
	/** The exact chunk text, equal to source.slice(startOffset, endOffset). */
	readonly text: string;
	/** Hash of the chunk bytes, recomputed and compared by SAT. */
	readonly contentHash: string;
}

/**
 * A single citation PAT attaches to its candidate answer. It is a claim that a
 * specific span of a specific source supports the answer. SAT treats every
 * field as untrusted until it re-reads the sealed bytes.
 */
export interface Citation {
	/** Chunk the proposer claims to be citing. */
	readonly chunkId: string;
	/** Source the chunk belongs to. */
	readonly sourceId: string;
	/** Byte offset (inclusive) the proposer claims to be quoting from. */
	readonly startOffset: number;
	/** Byte offset (exclusive) the proposer claims to be quoting to. */
	readonly endOffset: number;
	/** The quoted text the proposer attributes to the span above. */
	readonly quote: string;
}

/**
 * A candidate answer produced by PAT. `authorityDelta` is typed to the literal
 * `0`: a proposer can never emit a nonzero authority change, which encodes the
 * "no retrieved chunk may authorize an action" invariant at the type level.
 */
export interface CandidateAnswer {
	/** The drafted answer text, composed from retrieved chunks. */
	readonly text: string;
	/** Citations backing the answer, in the order PAT relied on them. */
	readonly citations: readonly Citation[];
	/** Always 0. A proposal never changes authority. */
	readonly authorityDelta: 0;
}

/** Why SAT rejected a single citation. */
export type CitationRejectionReason =
	| "unknown-chunk"
	| "source-mismatch"
	| "offset-out-of-range"
	| "offset-mismatch"
	| "hash-mismatch"
	| "quote-mismatch";

/** SAT's model-blind verdict for one citation. */
export interface CitationVerdict {
	readonly citation: Citation;
	readonly verified: boolean;
	/** Present only when `verified` is false. */
	readonly reason?: CitationRejectionReason;
}

/** The truth label DEMA shows for a result. */
export type TruthLabel = "verified" | "unverified" | "rejected";

/**
 * The labelled result DEMA returns to the caller. An answer is `rejected` when
 * any citation fails SAT, `unverified` when the proposal carries no citations,
 * and `verified` only when every citation is independently proven against the
 * sealed sources. The result is never stronger than its weakest cited evidence.
 */
export interface LabelledResult {
	readonly label: TruthLabel;
	/** True only when every citation was verified and at least one exists. */
	readonly accepted: boolean;
	/** The answer text, or an empty string when rejected. */
	readonly answer: string;
	/** Per-citation verdicts from SAT, in proposal order. */
	readonly verdicts: readonly CitationVerdict[];
	/** Mirrors the proposal's authority delta; always 0. */
	readonly authorityDelta: 0;
}
