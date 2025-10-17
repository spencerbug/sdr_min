# SDR Minimal Loop

A cognitive control loop modelled after cortical columns using Sparse Distributed Representations, built around the YCB-Habitat environment. The codebase focuses on explicit packet contracts, small interchangeable modules, and schema-validated message passing. Start with the blueprints below for the full architectural intent in this repository.


## 📘 Blueprints & References

- [System Blueprint](docs/design/design.md)
- [Packet Contracts](docs/design/packets.md)
- [Coding Paradigm](docs/design/style.md)
- [Examiner Walkthrough](docs/experiments/examiner.md)

## 🚀 Quickstart

### 1. Install dependencies

Use the provided Conda environment file to create an isolated toolchain (Python 3.9 + habitat-sim).

```bash
conda env create -f environment.yml
conda activate sdr-loop
```

### 2. Run the stub Examiner loop

Kick the loop with the default stub backend to verify the wiring. Pass `--backend habitat` later once the real adapter lands.

```bash
python examples/examiner_minimal.py --steps 120 --seed 7 --backend stub
```

Prefer a config file? Point `main.py` at a JSON payload and it will construct the same loop:

```bash
python main.py path/to/config.json
```

### 3. Run the tests

All packets and loop wiring are validated with `pytest`.

```bash
pytest
```

## 🧭 Repository Highlights

- `src/core/` hosts the environment adapter, sensor and context encoders, Hebbian maps, fusion/consensus layers, facet synthesizer, and the top-level loop.
- `contracts/` contains the draft-07 JSON Schemas used to validate every packet.
- `tests/` includes schema coverage, Hebbian/consensus unit tests, facet validation, and the 10-step smoke loop.

Read through `docs/design/design.md` for the long-term milestones, modules, and experiments laid out for the SDR system.
