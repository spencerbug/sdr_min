# Examiner Experiment

End-to-end recipe for running the Examiner loop. This covers both the default stub backend (contract tests) and the Habitat-backed path once assets are available.

## Prerequisites

- Conda environment created from `environment.yml`:

    ```bash
    conda env create -f environment.yml
    conda activate sdr-loop
    ```

- Packet schemas already vendored in `contracts/` (no action required).
- Optional (Habitat path): GPU-capable driver stack and Habitat assets.

## Assets

1. **YCB objects** — download once with the helper script:

   ```bash
   python scripts/download_ycb.py --dest assets/ycb
   ```

2. **Void examiner scene** — emit the black-shell GLB used by the real adapter:

   ```bash
   python scripts/generate_void_scene.py --dest assets/scenes/examiner/void_black.glb
   ```

All paths above match the defaults baked into `AssetConfig`.

## Running the Minimal Demo (stub backend)

The stub path exercises the loop wiring without Habitat dependencies. It remains the default in the example runner.

```bash
python examples/examiner_minimal.py --steps 120 --seed 7 --backend stub
```

Output is a JSON summary containing entropy, peakiness, average facet loss, and step count. Use `-h` to see container arguments.

## Running Against Habitat

Once assets are staged, switch the backend via the config schema. Create a JSON payload (example below) and run through `main.py`:

```json
{
  "steps": 240,
  "seed": 3,
  "env": {
    "backend": "habitat",
    "scenario": "examiner",
    "columns": ["col0"],
    "sensor": {
      "resolution": [96, 96],
      "modalities": ["rgb", "depth"],
      "hfov": 70.0
    },
    "assets": {
      "ycb_root": "assets/ycb",
      "scene_root": "assets/scenes",
      "examiner_scene": "examiner/void_black.glb"
    }
  }
}
```

When you are ready to invoke Habitat directly from the example runner, pass `--backend habitat`.

Execute the config:

```bash
python main.py path/to/examiner_habitat.json
```

The adapter will initialise Habitat-Sim, spawn a single YCB object at the origin, and emit RGB-D patches plus pose/context packets per tick.

## Observability & Logs

- `examples/examiner_minimal.py` prints summary metrics; add `--steps` or `--seed` to explore different horizons.
- For richer logging (CSV/JSONL, facet dumps), see the outstanding tasks in `docs/issues/examiner_demo.md`.

## Troubleshooting

- `ModuleNotFoundError: habitat_sim` — ensure the `sdr-loop` environment is active; re-run `conda env create` if necessary.
- `TypeError: 'bytes' object is not callable` when generating the scene – ensure you are on the latest commit where `generate_void_scene.py` uses `set_binary_blob`.
- Missing assets trigger `EnvAssetError`; confirm paths under `assets/` match the config.

## Next Steps

- Wire the real adapter into the loop (`EnvConfig.backend="habitat"`).
- Expand tests to cover both stub and Habitat modes (see `docs/issues/examiner_demo.md`).
