import { createHash } from "node:crypto";
import type { SealedChunk, SourceDocument } from "./types.ts";

/**
 * In-process lexical index with a sealed source store and a candidate graph.
 *
 * Problem: the governance spine needs a store that PAT can retrieve from and
 * that SAT can independently re-read, without any external cluster, OpenSearch,
 * or graph database. Example: ingest two short notes, then let PAT rank chunks
 * by TF-IDF while SAT re-reads the exact bytes a citation claims. Solution: a
 * single in-memory structure that holds (1) the sealed source bytes keyed by
 * id, (2) chunks with byte offsets and a content hash, (3) a TF-IDF lexical
 * index, and (4) a term co-occurrence adjacency used as a cheap candidate
 * graph for graph-walk expansion.
 *
 * "Sealed" here means the raw document text is stored verbatim and never
 * mutated after ingest; SAT reads from it to verify citations.
 */

/** Lowercase alphanumeric tokenization shared by ingest and query time. */
function tokenize(text: string): string[] {
	const matches = text.toLowerCase().match(/[a-z0-9]+/g);
	return matches ?? [];
}

function hashBytes(text: string): string {
	return createHash("sha256").update(text, "utf8").digest("hex");
}

interface ChunkStats {
	readonly chunk: SealedChunk;
	/** Raw term frequency per token within this chunk. */
	readonly termFrequency: Map<string, number>;
	/** Token count, used to normalize term frequency. */
	readonly length: number;
}

export interface ChunkScore {
	readonly chunk: SealedChunk;
	readonly score: number;
}

export interface IngestOptions {
	/** Target chunk size in characters. Defaults to 280. */
	readonly chunkSize?: number;
}

const DEFAULT_CHUNK_SIZE = 280;

/**
 * Splits a document into contiguous chunks on paragraph and sentence
 * boundaries, falling back to a hard character cap. Returns byte offsets so a
 * citation can be re-read exactly from the sealed source.
 */
function splitIntoSpans(text: string, chunkSize: number): Array<{ start: number; end: number }> {
	const spans: Array<{ start: number; end: number }> = [];
	const boundary = /\n{2,}|(?<=[.!?])\s+/g;
	let spanStart = 0;
	let cursor = 0;
	for (const match of text.matchAll(boundary)) {
		const breakAt = match.index + match[0].length;
		if (breakAt - spanStart >= chunkSize) {
			spans.push({ start: spanStart, end: breakAt });
			spanStart = breakAt;
		}
		cursor = breakAt;
	}
	if (spanStart < text.length) spans.push({ start: spanStart, end: text.length });
	else if (spans.length === 0 && text.length > 0) spans.push({ start: 0, end: text.length });

	// Enforce a hard cap so pathological inputs without boundaries still chunk.
	const capped: Array<{ start: number; end: number }> = [];
	for (const span of spans) {
		let start = span.start;
		while (span.end - start > chunkSize) {
			capped.push({ start, end: start + chunkSize });
			start += chunkSize;
		}
		if (span.end > start) capped.push({ start, end: span.end });
	}
	// Avoid an unused-variable lint on the running cursor used during scanning.
	void cursor;
	return capped;
}

export class KnowledgeIndex {
	private readonly sources = new Map<string, SourceDocument>();
	private readonly chunkStats = new Map<string, ChunkStats>();
	/** Number of chunks each term appears in, for inverse document frequency. */
	private readonly documentFrequency = new Map<string, number>();
	/** Candidate graph: term -> set of co-occurring terms within a chunk. */
	private readonly termAdjacency = new Map<string, Set<string>>();
	private chunkCounter = 0;

	/**
	 * Seals a source document and indexes its chunks. Re-ingesting the same id
	 * replaces the previous version wholesale. Returns the sealed chunks.
	 */
	ingest(document: SourceDocument, options: IngestOptions = {}): readonly SealedChunk[] {
		const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
		if (this.sources.has(document.id)) this.removeSource(document.id);
		this.sources.set(document.id, document);

		const produced: SealedChunk[] = [];
		for (const span of splitIntoSpans(document.text, chunkSize)) {
			const text = document.text.slice(span.start, span.end);
			if (text.trim().length === 0) continue;
			const chunk: SealedChunk = {
				id: `${document.id}#${this.chunkCounter++}`,
				sourceId: document.id,
				startOffset: span.start,
				endOffset: span.end,
				text,
				contentHash: hashBytes(text),
			};
			this.registerChunk(chunk);
			produced.push(chunk);
		}
		return produced;
	}

	private registerChunk(chunk: SealedChunk): void {
		const tokens = tokenize(chunk.text);
		const termFrequency = new Map<string, number>();
		for (const token of tokens) termFrequency.set(token, (termFrequency.get(token) ?? 0) + 1);
		this.chunkStats.set(chunk.id, { chunk, termFrequency, length: tokens.length });

		for (const term of termFrequency.keys()) {
			this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
			let neighbours = this.termAdjacency.get(term);
			if (!neighbours) {
				neighbours = new Set<string>();
				this.termAdjacency.set(term, neighbours);
			}
			for (const other of termFrequency.keys()) if (other !== term) neighbours.add(other);
		}
	}

	private removeSource(sourceId: string): void {
		this.sources.delete(sourceId);
		for (const [id, stats] of this.chunkStats) {
			if (stats.chunk.sourceId !== sourceId) continue;
			for (const term of stats.termFrequency.keys()) {
				const next = (this.documentFrequency.get(term) ?? 1) - 1;
				if (next <= 0) {
					this.documentFrequency.delete(term);
					this.termAdjacency.delete(term);
				} else {
					this.documentFrequency.set(term, next);
				}
			}
			this.chunkStats.delete(id);
		}
	}

	/** Total number of indexed chunks. */
	get chunkCount(): number {
		return this.chunkStats.size;
	}

	/** Returns the sealed source document, or undefined if never ingested. */
	getSource(sourceId: string): SourceDocument | undefined {
		return this.sources.get(sourceId);
	}

	/** Returns the sealed chunk by id, or undefined if unknown. */
	getChunk(chunkId: string): SealedChunk | undefined {
		return this.chunkStats.get(chunkId)?.chunk;
	}

	/**
	 * Ranks chunks against the query with TF-IDF cosine-like scoring. Returns
	 * the top `limit` scoring chunks in descending order.
	 */
	lexicalSearch(query: string, limit: number): ChunkScore[] {
		const queryTerms = tokenize(query);
		if (queryTerms.length === 0 || this.chunkStats.size === 0) return [];
		const totalChunks = this.chunkStats.size;
		const scored: ChunkScore[] = [];
		for (const stats of this.chunkStats.values()) {
			let score = 0;
			for (const term of queryTerms) {
				const tf = stats.termFrequency.get(term);
				if (!tf || stats.length === 0) continue;
				const df = this.documentFrequency.get(term) ?? 0;
				if (df === 0) continue;
				const idf = Math.log((1 + totalChunks) / (1 + df)) + 1;
				score += (tf / stats.length) * idf;
			}
			if (score > 0) scored.push({ chunk: stats.chunk, score });
		}
		scored.sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id));
		return scored.slice(0, limit);
	}

	/**
	 * Expands a set of query terms by one hop over the candidate graph, so a
	 * graph walk can surface chunks that share co-occurring terms even when they
	 * do not contain the original query terms directly.
	 */
	expandTerms(query: string): Set<string> {
		const expanded = new Set<string>();
		for (const term of tokenize(query)) {
			expanded.add(term);
			const neighbours = this.termAdjacency.get(term);
			if (neighbours) for (const neighbour of neighbours) expanded.add(neighbour);
		}
		return expanded;
	}
}
