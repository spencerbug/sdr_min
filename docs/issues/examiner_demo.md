# Examiner Demo Progress

Use this list to track outstanding work for a fully featured Examiner demo. Items checked off are already implemented on `master`.

## ✅ Completed

- [x] Ship a static `ContextGates` helper so the Examiner loop defaults to all-ones gains.
- [x] Provide `examples/examiner_minimal.py` as the one-command demo runner.
- [x] Refresh README/docs with the examiner workflow and backend toggle.
- [x] Unify runner wiring on the shared `EnvConfig` schema and document the defaults.
- [x] Expose a `--backend` flag on the runner so stub vs Habitat can be selected at launch.
- [x] Add helper scripts to download YCB assets and generate the black void scene.
- [x] Implement the Habitat-backed Examiner adapter with Habitat-Sim orbit control and packet emission.

## ⏳ Remaining

1. **Pipe step metrics to lightweight logs.** Capture entropy, peakiness, and action counts each tick (CSV/JSONL) so runs can be compared without a debugger.
2. **Dump sample facet reconstructions for inspection.** Save the top shared facet predictions (e.g., PNG/NPZ) after each run to validate facet plumbing end-to-end.
3. **Add an integration test around the example runner.** Extend the smoke suite with a pytest that executes the example for a short horizon and asserts deterministic summary stats/log artefacts (skip Habitat when assets are unavailable).
