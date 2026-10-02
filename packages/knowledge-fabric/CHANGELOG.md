# Changelog

## [Unreleased]

### Added

- Added the isolated knowledge fabric package with a governance-split retrieval spine: in-process ingest and sealed source store, a local TF-IDF lexical index with a term co-occurrence candidate graph, a PAT proposer that drafts cited answers with `authorityDelta` fixed to `0`, a model-blind SAT verifier that re-checks every citation against the sealed source bytes, and a DEMA surface that returns labelled results. Includes the red-first acceptance tests that a forged or unprovable citation is rejected and that no retrieved chunk may authorize an action.
