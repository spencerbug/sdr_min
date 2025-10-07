Here’s a draft of a **style.md** you could drop straight into a repo. It’s written as a paradigm/style document for purpose-built, AI-maintainable, well-tested, concise, human-readable software.

---

# STYLE.md

### A Paradigm for Purpose-Built, AI-Maintainable, Human-Readable Code

This document describes the style and principles for writing software in this repository. The goal: **concise, testable, human-friendly code that other humans *and* AI assistants can read, extend, and verify.**

---

## 1. Core Principles

* **Conciseness over generality.** Write *just enough* abstraction for the current problem. Do not add knobs for hypothetical future needs.
* **Data contracts first.** Define clear data packets or schemas (JSON Schema, dataclasses, Pydantic). All modules communicate via these contracts.
* **Documentation is executable.** Every major component has:

  * A short English description
  * Type hints and shape annotations
  * Example usage or JSON sample
* **Tests are specs.** Unit tests + golden files serve as living documentation for expected behavior.

---

## 2. Structure & Layout

* Each module ≤ ~200 lines where possible. Small, focused, composable.
* Standard repo skeleton:

  ```
  core/            # environment, learner, policy, sensors, etc.
  contracts/       # JSON schemas for packets
  examples/        # runnable demo loops, <200 lines
  tests/           # unit tests, golden snapshots
  docs/            # README, PACKETS.md, DESIGN.md
  ```
* One **golden-path example** showing end-to-end loop must always work (`python examples/minimal_loop.py`).

---

## 3. Code Style

* **Type hints everywhere.** Use `numpy.typing.NDArray[np.float32]` etc.
* **Docstrings with shapes.**

  ```python
  def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
      """
      Args:
        scores: (N,) float32 array.
        k: number to select.

      Returns:
        idx: (k,) int32 array sorted by score descending.
      """
  ```
* **Deterministic seeds**: always pass seed into constructors; avoid hidden randomness.
* **No nested configs.** Wire modules in code, not YAML. Let Python be the DSL.
* **One owner of time, one owner of action.** Explicit control loop, no hidden lifecycles.

---

## 4. Documentation Standards

* **README.md:** installation + 10-line run example.
* **PACKETS.md:** each packet with description, shape table, JSON sample, schema link.
* **DESIGN.md:** invariants and glossary (e.g., sparsity %, row Top-K, entropy).
* **CHANGELOG.md:** packet versioning (`belief.v1 → belief.v2`).

---

## 5. Testing Guidelines

* **Unit tests per pure function.** One property per test.
* **Schema validation:** every packet validated against JSON Schema.
* **Golden files:** snapshot outputs (JSON, CSV) for deterministic runs; compared byte-for-byte in CI.
* **Smoke test:** 100-step deterministic run; ensures the core loop produces consistent packets.
* Tests must run in <10 seconds.

---

## 6. Human & AI Affordances

* **Clear glossary**: define project-specific terms once, reuse consistently.
* **Explicit names**: avoid acronyms unless defined in docs.
* **Small, composable files**: easy for AI assistants to regenerate or extend without context loss.
* **Versioned packets**: append `.v1`, `.v2` to schemas; never silently change shape.
* **Stable interfaces, swappable internals**: loop structure and packets remain constant; algorithms can be replaced freely.

---

## 7. Anti-Patterns to Avoid

* Nested configuration trees pretending to be code.
* Wide inheritance hierarchies with abstract base classes.
* Hidden state transitions (implicit step phases, double ticks).
* “Flexibility” that adds more adapters than the problem requires.
* Coupling tests to internal state instead of packet I/O.

---

## 8. Philosophy

The style here treats **code as a minimal framework**:

* Concise for humans to read in one sitting.
* Structured so AI tools can regenerate, extend, and test reliably.
* Versioned contracts so evolution is safe.

We favor **clarity and correctness now** over speculative generality. Frameworks may come and go; packet contracts and tests endure.

---

Would you like me to also generate a **boilerplate `PACKETS.md`** in the same style so you have both the coding paradigm and a packet documentation scaffold ready to go?
