import { createHash } from "node:crypto";
import type { KnowledgeIndex } from "./index-store.ts";
import type { CandidateAnswer, Citation, CitationVerdict } from "./types.ts";

/**
 * SAT: the independent verifier. For every citation PAT attached, SAT re-reads
 * the sealed source bytes and proves the citation or rejects it. SAT is
 * model-blind: it never sees how the answer was drafted, only the citation's
 * claimed source, offsets, and quote, which it checks against the sealed store.
 *
 * Problem: a proposer can fabricate a citation (wrong offsets, a quote that was
 * never in the source, or a chunk id that does not exist). Example: PAT claims
 * `source "notes" [0,10)` says "APPROVED" when the sealed bytes say nothing of
 * the sort. Solution: SAT recomputes everything from the sealed source. A
 * citation is verified only when the chunk exists, its offsets and content hash
 * still match the sealed bytes, and the quoted text equals the sealed slice.
 */

function hashBytes(text: string): string {
	return createHash("sha256").update(text, "utf8").digest("hex");
}

export class VerifierAgent {
	private readonly index: KnowledgeIndex;

	constructor(index: KnowledgeIndex) {
		this.index = index;
	}

	/** Verifies a single citation against the sealed source bytes. */
	verifyCitation(citation: Citation): CitationVerdict {
		const chunk = this.index.getChunk(citation.chunkId);
		if (!chunk) return { citation, verified: false, reason: "unknown-chunk" };
		if (chunk.sourceId !== citation.sourceId) return { citation, verified: false, reason: "source-mismatch" };
		if (chunk.startOffset !== citation.startOffset || chunk.endOffset !== citation.endOffset) {
			return { citation, verified: false, reason: "offset-mismatch" };
		}

		const source = this.index.getSource(citation.sourceId);
		if (!source) return { citation, verified: false, reason: "source-mismatch" };
		if (
			citation.startOffset < 0 ||
			citation.endOffset > source.text.length ||
			citation.startOffset >= citation.endOffset
		) {
			return { citation, verified: false, reason: "offset-out-of-range" };
		}

		const sealedSlice = source.text.slice(citation.startOffset, citation.endOffset);
		if (hashBytes(sealedSlice) !== chunk.contentHash) return { citation, verified: false, reason: "hash-mismatch" };
		if (sealedSlice !== citation.quote) return { citation, verified: false, reason: "quote-mismatch" };

		return { citation, verified: true };
	}

	/** Verifies every citation on a candidate answer, preserving order. */
	verify(candidate: CandidateAnswer): CitationVerdict[] {
		return candidate.citations.map((citation) => this.verifyCitation(citation));
	}
}
