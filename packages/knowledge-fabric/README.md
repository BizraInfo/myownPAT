# @earendil-works/pi-knowledge-fabric

An isolated, framework-agnostic knowledge fabric. It is not a single retrieval
algorithm; it is a governance split mapped onto a propose/verify boundary:

```
ingest -> local lexical index -> PAT propose -> SAT verify -> DEMA label
```

This package depends on no other workspace package and is not wired into the
coding agent or any Pi runtime. It exposes a thin public API: ingest a source,
query it, and read back a labelled result.

## The governance roles

- **DEMA** is the human-facing surface. Intent in, labelled answer out. It is
  the query/answer boundary and the presentation layer. DEMA never upgrades a
  label beyond what the verifier proved.
- **PAT** is the proposer. It does the retrieval (TF-IDF lexical ranking plus a
  one-hop walk over a term co-occurrence candidate graph) and composes a
  candidate cited answer. PAT may only propose: every candidate it emits carries
  `authorityDelta === 0`, a field typed to the literal `0`.
- **SAT** is the independent verifier. It re-reads the sealed source bytes for
  every citation PAT used and rejects the answer if any citation is forged or
  unprovable. SAT is model-blind: it never sees how the answer was drafted, only
  the citation's claimed source, offsets, and quote, which it checks against the
  sealed store.

## Invariants

- A forged or unprovable citation causes SAT to reject the answer.
- No retrieved chunk may ever authorize an action. A chunk is evidence, never an
  authority grant. `authorityDelta` is `0` on both proposals and results.
- An answer is never stronger than its weakest cited evidence: a single failed
  citation rejects the whole answer.

## Usage

```ts
import { KnowledgeFabric } from "@earendil-works/pi-knowledge-fabric";

const fabric = new KnowledgeFabric();
fabric.ingest({
	id: "notes",
	title: "Deployment notes",
	text: "The staging rollout finished cleanly on Tuesday. Latency stayed flat.",
});

const result = fabric.query("rollout latency");
// result.label is "verified" | "unverified" | "rejected"
// result.accepted is true only when every citation proved out
// result.answer carries the cited text, empty when rejected
// result.authorityDelta is always 0
```

### Lower-level agents

The individual agents are exported for callers who want to drive the split
directly:

```ts
import { KnowledgeIndex, ProposerAgent, VerifierAgent } from "@earendil-works/pi-knowledge-fabric";

const index = new KnowledgeIndex();
index.ingest({ id: "doc", text: "..." });

const candidate = new ProposerAgent(index).propose("query");
const verdicts = new VerifierAgent(index).verify(candidate);
```

## Scope of the first slice

- One source type: in-memory UTF-8 text documents.
- A local, in-process index only. No external cluster, OpenSearch, or graph
  database. Promotion to external stores is deferred behind a measured
  threshold, not committed here.
- Each later source type or store is an increment behind the same proven
  propose/verify gate.

## Development

```sh
npm run build   # tsc -p tsconfig.build.json
npm test        # vitest --run
```
