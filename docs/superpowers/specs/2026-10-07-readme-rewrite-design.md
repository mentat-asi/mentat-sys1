# Mentat Sys1 README Rewrite Design

## Goal

Make the GitHub README present Mentat Sys1 v0.2 as the current product in one
clear narrative, while preserving v0.1 as a supported frozen release.

## Audience

- Engineers evaluating an open-weights System One decision model
- Benchmark reviewers checking JevBench claims
- Users who need to run either v0.2 or v0.1 locally

## Narrative

The first screen should answer three questions:

1. What is Mentat Sys1?
2. What improved in v0.2?
3. How can I run it?

The core progression is:

`SemIF 187/231 -> Mentat v0.1 200/231 -> Mentat v0.2 206/231`

The README must describe `206/231` as a local Public-231 result, not an
official composite leaderboard rank.

## Structure

1. Product name and one-sentence positioning
2. Current v0.2 result and release links
3. Compact progress and selected 4B comparison tables
4. Short explanation of the decision API and supported field types
5. v0.2 quick start
6. Minimal request and response example using `mentat-sys1-v0.2`
7. Reproducibility evidence from the new runtime rerun
8. Compact v0.1 compatibility note
9. Links to the model card, reproducibility guide, releases, and license

## Content Rules

- Keep v0.2 as the default model throughout the main flow.
- Keep v0.1 details out of the primary quick start and API example.
- Retain only comparison models whose published Public-231 values are already
  documented in the repository.
- State the exact local evaluation scope and avoid unsupported ranking claims.
- Include the latest A800 rerun:
  - v0.1: 200/231, 231/231 valid, zero errors
  - v0.2: 206/231, 231/231 valid, zero errors
  - both runs reproduced their historical predictions exactly
- Keep commands copy-pasteable and preserve pinned model revisions.
- Use restrained Markdown with no decorative badges or screenshots.

## Removed From The Main Flow

- The full v0.1 leaderboard narrative
- Repeated explanations of model package contents
- Detailed contamination methodology
- Long calibration and latency discussion
- Duplicate serving instructions

These details remain available in the existing model card, reproducibility
guide, release notes, and archived v0.1 bench request.

## Validation

- Every primary command uses the version-neutral GitHub repository.
- Every primary request uses `mentat-sys1-v0.2`.
- Model IDs, revisions, hashes, and JevBench totals match the frozen configs
  and the latest A800 rerun.
- Markdown links resolve to intended repository documents.
- Existing test, Ruff, mypy, and package-build gates remain unchanged.
