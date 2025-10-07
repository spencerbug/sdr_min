# SDR Minimal Loop

A compact spike-driven representation (SDR) cortical column loop built around the YCB-Habitat environment. The codebase focuses on explicit packet contracts, small interchangeable modules, and schema-validated message passing. Start with the blueprints below for the full architectural intent in this repository.

## 📘 Blueprints & References

- [System Blueprint](docs/design.md)
- [Packet Contracts](docs/packets.md)
- [Coding Paradigm](docs/style.md)

## 🚀 Quickstart

### 1. Install dependencies

Use the provided Conda environment file to create an isolated toolchain (Python 3.9 + habitat-sim).

```bash
conda env create -f environment.yml
conda activate sdr-loop
```

### 2. Run the smoke loop

Execute the default experiment configuration via the CLI entry point. Supply a JSON config file if you need to override defaults.

```bash
python main.py
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

Read through `docs/design.md` for the long-term milestones, modules, and experiments laid out for the SDR system.
