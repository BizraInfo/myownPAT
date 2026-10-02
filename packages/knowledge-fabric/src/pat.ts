import type { KnowledgeIndex } from "./index-store.ts";
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
}

const DEFAULT_MAX_CHUNKS = 3;

export class ProposerAgent {
	private readonly index: KnowledgeIndex;

	constructor(index: KnowledgeIndex) {
		this.index = index;
	}

	propose(query: string, options: ProposeOptions = {}): CandidateAnswer {
		const maxChunks = options.maxChunks ?? DEFAULT_MAX_CHUNKS;
		const useGraphWalk = options.useGraphWalk ?? true;

		const effectiveQuery = useGraphWalk ? [...this.index.expandTerms(query)].join(" ") : query;
		const ranked = this.index.lexicalSearch(effectiveQuery, maxChunks);

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
