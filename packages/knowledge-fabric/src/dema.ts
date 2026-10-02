import { Value } from "typebox/value";
import { type IngestOptions, KnowledgeIndex } from "./index-store.ts";
import { type ProposeOptions, ProposerAgent } from "./pat.ts";
import { VerifierAgent } from "./sat.ts";
import type { CitationVerdict, LabelledResult, SealedChunk, TruthLabel } from "./types.ts";
import { type SourceDocument, sourceDocumentSchema } from "./types.ts";

/**
 * DEMA: the human-facing surface. Query intent in, labelled result out. DEMA
 * wires the governance split together: it asks PAT to propose, hands the
 * proposal to SAT to verify, and composes the truth label. It never bypasses
 * SAT, and it never upgrades a label beyond what SAT proved.
 *
 * The standing closure invariant lives here too: the returned result always
 * reports `authorityDelta: 0`, so a retrieved chunk can never be surfaced as an
 * actionable authority grant. An answer is never stronger than its weakest
 * cited evidence: a single failed citation rejects the whole answer.
 */

function labelFor(verdicts: readonly CitationVerdict[]): { label: TruthLabel; accepted: boolean } {
	if (verdicts.length === 0) return { label: "unverified", accepted: false };
	const allVerified = verdicts.every((verdict) => verdict.verified);
	if (allVerified) return { label: "verified", accepted: true };
	return { label: "rejected", accepted: false };
}

export interface KnowledgeFabricOptions {
	readonly propose?: ProposeOptions;
}

export class KnowledgeFabric {
	private readonly index: KnowledgeIndex;
	private readonly proposer: ProposerAgent;
	private readonly verifier: VerifierAgent;

	constructor(index: KnowledgeIndex = new KnowledgeIndex()) {
		this.index = index;
		this.proposer = new ProposerAgent(index);
		this.verifier = new VerifierAgent(index);
	}

	/** Validates and seals a source document, returning its sealed chunks. */
	ingest(document: SourceDocument, options: IngestOptions = {}): readonly SealedChunk[] {
		const parsed = Value.Parse(sourceDocumentSchema, document);
		return this.index.ingest(parsed, options);
	}

	/**
	 * Runs the full spine: PAT proposes a cited answer, SAT verifies every
	 * citation against the sealed sources, and DEMA returns the labelled result.
	 */
	query(intent: string, options: KnowledgeFabricOptions = {}): LabelledResult {
		const candidate = this.proposer.propose(intent, options.propose);
		const verdicts = this.verifier.verify(candidate);
		const { label, accepted } = labelFor(verdicts);
		return {
			label,
			accepted,
			answer: accepted ? candidate.text : "",
			verdicts,
			authorityDelta: 0,
		};
	}
}
