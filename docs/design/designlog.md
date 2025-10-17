# Oct 2025
---

### 1. **Foundational Core**

* Defined the minimal **SDR learning module** and **YCB-Habitat environment blueprint**.
* Established the six core associative maps (A_FP, A_PF, A_PP, A_FF, A_CP, A_CF) and Hebbian/Top-K update rule.
* Built consensus (Product-of-Experts) logic, with entropy-weighted fusion across columns.
* Introduced **BeliefPacket** and **Facet reconstruction** contracts — clean JSON schemas for modularity.

### 2. **Environment Design**

* Finalized **Examiner**, **Explorer**, and **Goalseeker** scenarios.
* Decided on *Examiner first*: single object, pan/tilt camera on 2D chart.
* Established column wiring templates:

  * **Examiner:** one column (RGB-D → f_SDR, pan/tilt velocity → g_SDR).
  * **Explorer:** two columns (locomotion + eye).
  * **Goalseeker:** future extension with arm joints and tactile feedback.
* Agreed that **local proprioception** flows into phase inputs, **global proprioception & IMU** join the context bus.

### 3. **Control Loops Evolution**

* Introduced **effector–effect** and **feature–phase** control loops:

  * First-order: direct phase correction (servo-like).
  * Second-order: feature-counterfactual correction.
  * Future: sequence/time-predictive loops.
* Converged on the idea that lower-level proprioceptive columns can use direct phase loops for low-latency control; higher-level feature columns use counterfactual loops.

### 4. **Policy Layer Clarification**

* Split policies by scenario:

  * *Examiner*: mixed discrete (switch/jump) + continuous (pan/tilt).
  * *Explorer*: continuous only (drive + eye).
  * *Goalseeker*: multi-joint continuous, plus reward.
* Replaced “roll logits” with a proper **utility function** that blends entropy reduction and exploration.
* Defined **effector-effect servo** and **discrete entropy manager** policy types.

### 5. **DSL & Wiring YAML**

* Designed a clean YAML DSL that declares:

  * Sensors, motors, columns, buses, control loops, consensus groups.
  * Explicit wiring between signals and actions.
  * Validation rules for dimensionality, signal existence, and bus typing.
* Each experiment becomes a single YAML file; no more nested config sprawl.
* Added adaptive-scaling option in fusion for context/proprioception weighting.

### 6. **Logging System**

* Added declarative logging spec with sinks, streams, triggers, and transforms.
* Supports scalars (CSV), arrays (NPZ/Zarr), events (JSONL), images, and videos.
* Enables lightweight, experiment-level observability.

### 7. **Conceptual Expansions**

* Discussed adding **head-direction**, **salience**, **surprise**, and **attention gating**—but deferred as out of scope for the Examiner MVP.
* Explored the theoretical **effector–effect gap** as a unifying way to model sensorimotor learning.
* Recognized possible abstraction layer: a **Control Loop DSL** that could eventually describe wiring, sensors, motors, and adaptive loops in one file.

### 8. **Transition Away from Monty Framework**

* Concluded that the Monty framework was too heavy and rigid for rapid experimental iteration.
* Spent excessive time fighting with framework abstractions instead of focusing on core design and scientific goals.
* Decided to build an **anti-framework** — a clean-slate architecture prioritizing:
  - **Conciseness over generality.**
  - **Data contracts first**, enforcing clear JSON/YAML schemas before implementation.
  - **Explicit blueprints** written out in markdown prior to any code.
  - **Readable, testable code** optimized for both human and AI-assisted comprehension.
  - **No hidden dependencies** or deep inheritance; favor explicit composition.
* Wrote the **Style Guide for Anti-Framework Development**, covering:
  - Naming conventions and self-documenting functions.
  - Rules for code minimalism, determinism, and reproducibility.
  - Documentation standards ensuring that every module’s purpose, data flow, and side effects are clearly explained.
  - Principles of *“code as contract”* — treating data interfaces as the primary boundary of modularity.
* The philosophy: “If an AI can’t read, verify, and extend your code with confidence, it’s too opaque for humans too.”

---

### 9. **Launch of SDR_MIN Project**

* Created a new repository: [sdr_min](https://github.com/spencerbug/sdr_min)
* Goal: implement a minimal, fully transparent, and self-contained version of the SDR-based cortical model.

Core actions taken:
- Wrote detailed **blueprints** for every subsystem:
  - Core learning module (SDR-based associative learner).
  - Environment and scenario specification.
  - Motor and policy system.
  - Logging, visualization, and data I/O layers.
- Defined **data contracts** (schemas) for:
  - Observations, actions, SDRs, associative matrices, and belief states.
  - Logging events and metrics.
- Each subsystem was documented as a standalone, importable module with clear test entry points.
- Adopted a **functional-first** design: minimal class hierarchies, pure functions where possible.
- Added reproducible experiment templates to demonstrate single-column learning and counterfactual reconstruction.

---

### 10. **Foundational Core**

* Defined the minimal **SDR learning module** and **YCB-Habitat environment blueprint**.
* Established six core associative maps: `A_FP`, `A_PF`, `A_PP`, `A_FF`, `A_CP`, `A_CF`.
* Standardized Hebbian/Top-K update rules and sparse row pruning.
* Implemented consensus (Product-of-Experts) fusion with entropy weighting.
* Introduced `BeliefPacket` and `FacetReconstruction` JSON contracts for plug-and-play modularity.

---

### 11. **Environment Design**

* Finalized three experiment archetypes:
  - **Examiner:** single object, pan/tilt view control.
  - **Explorer:** two columns (locomotion + eye).
  - **Goalseeker:** multi-effector future extension.
* Chose *Examiner* as the MVP scenario:
  - Single-column.
  - Pan/tilt control on a 2D chart of object views.
  - Each patch represented as an RGB-D observation projected to feature SDRs.
* Established data flow conventions:
  - Local proprioception → phase inputs.
  - Global proprioception (IMU, rotation) → context bus.

---

### 12. **Control Loop Evolution**

* Identified two key control feedback loops:
  - **Effector–effect loop** (phase correction).
  - **Feature–phase counterfactual loop** (predictive correction).
* Defined future third layer: **temporal/predictive** control for sequence learning.
* Established hierarchical control principle:
  - Lower columns: proprioceptive phase correction.
  - Higher columns: predictive feature-based control.
* Introduced the **effector–effect gap** as a unified abstraction for learning causal mappings.

---

### 13. **Policy Layer Clarification**

* Split policy strategies by experimental scenario:
  - *Examiner:* discrete + continuous (pan/tilt and jump actions).
  - *Explorer:* continuous (locomotion, gaze).
  - *Goalseeker:* continuous multi-joint control.
* Replaced “roll logits” shortcut with full **utility function**:
  - Entropy reduction term.
  - Loop-closure bonus.
  - Action cost and exploration bias.
* Defined **Effector-Effect Servo** (continuous controller) and **Entropy Manager** (discrete action selector).

---

### 14. **DSL & Wiring YAML**

* Designed a concise YAML-based **DSL** for declaring:
  - Sensors, motors, columns, buses, control loops, and fusion groups.
  - Wiring between signals and buses.
* Introduced validation rules:
  - Dimensionality checks.
  - Bus existence and typing.
  - Parameter range enforcement.
* Each experiment is fully declarative — no deep config inheritance.
* Added adaptive fusion weights for dynamic context or proprioceptive influence.

---

### 15. **Logging System**

* Designed a **declarative logging spec**:
  - Defines sinks (CSV, NPZ, JSONL, image/video).
  - Triggers (per step, per event).
  - Transform pipelines for computed metrics.
* Supports:
  - Scalars (entropy, peakiness).
  - Arrays (SDRs, logits).
  - Visualizations (counterfactual canvases).
* Enables rich observability without bloated telemetry frameworks.

---

### 16. **Conceptual Expansions**

* Deferred features:
  - Head direction, salience, attention gating.
  - Temporal pooling and behavior modeling.
* Defined abstraction for future **Control Loop DSL** describing feedback structures and adaptive wiring.
* Introduced **surprise signal** as a potential context-gating mechanism.
* Clarified theoretical roles of context and proprioception as stabilizing priors.

---

### Summary

This transition marks a full pivot from framework-driven development (Monty) to **principle-driven architecture** (sdr_min):
- Every system starts with a blueprint, data contract, and wiring schema.
- Code is minimal, deterministic, and human-readable.
- The entire stack — from sensors to policies to learning — is transparent and testable.
- SDR_MIN becomes a sandbox for experimenting with biological learning principles in a way that’s accessible to both human collaborators and AI coding agents.


---

## 🧩 Current Focus

**Goal:** Deliver the *Examiner* MVP — one column, one object, pan/tilt exploration loop.

**Milestones:**

1. ✅ Finalize YAML schema & validator.
2. 🟡 Implement runtime loop (env → sensor → column → consensus → policy → env).
3. 🔲 Add minimal logger (entropy + Top-K snapshot).
4. 🔲 Run smoke test (100 steps, entropy curve sanity check).
5. 🔲 Render a 2D facet reconstruction for one object.

**Out-of-Scope for MVP (defer to v2+):**

* Multi-column Explorer setup.
* Arm & tactile columns (Goalseeker).
* Salience/motivation modules.
* Head-direction and sequence prediction loops.
* Full Control Loop DSL abstraction.


