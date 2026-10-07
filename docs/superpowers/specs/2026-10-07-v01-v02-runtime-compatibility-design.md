# Mentat Sys1 v0.1/v0.2 Runtime Compatibility Design

## Goal

Evolve the public repository into a version-neutral `mentat-sys1` runtime
that loads both frozen Mentat model releases:

- `yunqu/mentat-sys1-v0.1`
- `yunqu/mentat-sys1-v0.2`

The v0.1 release contract must remain reproducible. The v0.2 release must use
its own identity, artifact hashes, evaluation gate, and type-specific
calibration.

## Compatibility Contract

- Keep the Python import package and CLI command as `mentat_sys1` and
  `mentat-sys1`.
- Accept only the registered model IDs `mentat-sys1-v0.1` and
  `mentat-sys1-v0.2`.
- Derive the active model identity from the validated project configuration
  and require the model manifest to match it.
- Make model identity an instance property of the loaded backend. Do not use a
  mutable process-global active model.
- Keep the existing v0.1 configuration unchanged apart from schema
  generalization required to parse both registered releases.
- Add a v0.2 configuration with its frozen base revision, adapter/readout
  hashes, and Public-231 result of 206/231 with 48/48 Easy.
- Preserve all existing v0.1 tags. Rename the GitHub repository from
  `mentat-sys1-v0.1` to `mentat-sys1`; GitHub's redirect preserves old URLs.

## Calibration

The shared calibration loader supports exactly two schemas:

- Schema 1, `scalar_temperature`: validate one finite positive temperature and
  apply it to every decision type.
- Schema 2, `temperature_by_type`: validate a finite positive pooled
  temperature and an exact `choice`, `noul`, and `score` temperature map.

The backend selects the temperature for each request field by its decision
type. Unsupported schemas, missing types, extra types, booleans, non-finite
values, and non-positive values fail before model inference. Calibration
metadata returned by verification includes the pooled temperature, the full
type map, the calibration digest, and a schema-qualified version string.

## Release Verification

Artifact verification remains fail-closed:

- Configuration, model manifest, adapter metadata, tensor hashes, and readout
  shape must agree.
- Calibration payload and manifest records must agree with the selected model.
- Evaluation evidence is checked against the release-specific exact totals in
  configuration rather than a v0.1-only type literal.
- Existing v0.1 scalar-calibration and 200/231 tests remain green.
- New tests cover v0.2 identity, 206/231 gating, type-specific temperature
  selection, invalid schema rejection, and cross-version artifact rejection.

The public runtime does not include training data, training code, checkpoints,
optimizer state, or base-model weights.

## Documentation And Packaging

- Present the repository and Python distribution as `mentat-sys1` version
  `0.2.0`.
- Keep one quick-start path per model release with pinned Git and Hugging Face
  revisions.
- Make v0.2 the current release in the top-level README while retaining a clear
  v0.1 compatibility section.
- Keep model-specific historical evidence in versioned documentation instead
  of rewriting v0.1 evidence as v0.2.
- Update the lockfile and build metadata without renaming the Python module.

## Verification And Publication

Before remote changes:

1. Run the focused red/green tests for each new behavior.
2. Run the complete pytest suite, Ruff, strict mypy, frozen dependency sync,
   and wheel/sdist build.
3. Verify freshly downloaded, immutable v0.1 and v0.2 Hugging Face model
   packages against their matching configurations.
4. Confirm each model rejects the other version's configuration.
5. Confirm the working tree contains only intended public-runtime changes and
   no secrets or local paths.

Only after all gates pass:

1. Rename the GitHub repository to `mentat-sys1`.
2. Push the reviewed commit to `main`.
3. Create and push the immutable `v0.2.0` tag.
4. Verify the new URL, the old redirect, the tag, and a fresh clone.
