---

# `DESIGN.md`

# Multi-Column SDR Learning Module with Context SDR

A compact blueprint for a from-scratch SDR system that runs inside a simple YCB-Habitat loop:

1. Load experiment config
2. YCB Habitat spawns a random object at a random view
3. A column (or multiple) looks around on a 2D torus/chart $(u,v)$
4. Continue exploring or switch objects
5. Reconstruct a local **2.5D facet** via counterfactual completion
6. Compare predicted facet(s) to sensor renderings / ground-truth meshes

The design is minimal: explicit packets, product-of-experts logits, Hebbian maps with row-Top-K, and concise tests.

---

## Minimal File Structure (no framework dependencies)

```
sdr_min/
   core/
      assoc.py            # sparse Hebbian maps + row-Top-K pruning
      belief.py           # BeliefPacket helpers (entropy, peakiness, top-k)
      consensus.py        # entropy-weighted product-of-experts fusion
      context.py          # context SDR plumbing (stub)
      encoders.py         # SDR encoders + prototype bank stubs
      env_ycb.py          # Habitat-lite environment surface (stub)
      facet.py            # counterfactual facet synthesizer stub
      fusion.py           # column-level fusion helpers
      loop.py             # orchestrates environment ↔ SDR ↔ policy (WIP)
      phase.py            # path-integration helpers (stub)
      policy.py           # random policy stub emitting ActionMessage
      sensor.py           # patch → feature vector adapter
   contracts/
      action.schema.json
      belief.schema.json
      context.schema.json
      eval.schema.json
      facet.schema.json
      observation.schema.json
      pose.schema.json
   docs/
      DESIGN.md           # this blueprint
      ENVIRONMENT.md      # environment wiring notes
      PACKETS.md          # packet examples + schema pointers
      POLICY.md           # policy modes (current + planned)
      STYLE.md            # coding/style paradigm
   examples/              # (planned) minimal end-to-end loop scripts
   main.py               # tiny entry point / smoke harness
   tests/
      test_assoc_hebbian.py
      test_consensus_ctx.py
      test_facet_recon.py
      test_loop_smoke.py
      test_packets_schema.py
      # (planned) test_pi_wraparound.py, test_scene_load.py, ...
```

---

## Core Representations & Notation

### Feature SDRs

* Dim. $D$, binary, $k \approx 0.02D$ active; $f_{\text{SDR}}\in\{0,1\}^D$.

### Phase SDRs (allocentric)

* $M$ grid modules, module $m$ has $B_m$ cells, total $P=\sum_m B_m$.
* Continuous phase $\boldsymbol\phi_m\in[0,1)^2$; discretized to a bump per module.
* $g_{\text{SDR}}\in\{0,1\}^P$ is module-wise Top-K concatenation.

### Context SDRs

* Dim. $C$, $k_c \approx 0.02C$ active; $c_{\text{SDR}}\in\{0,1\}^C$.
* Sources vary by scenario (see `ENVIRONMENT.md` §6): Examiner carries morphological/non-morphological descriptors + switch pulse, Explorer adds head-direction, motion-rate, intent, and metronome bits, while Goalseeker further appends reward-state encodings.

---

## Scenario-Specific Column Wiring

We maintain 2D phase states per column; each column also commands its own 2D motor primitive

### Examiner — Single Column (Object Inspection)

* **Features**: SDR-encoded RGB-D patch / local point cloud.
* **Phase input**: $(v_u, v_v)$ pan/tilt velocity derived from torus deltas.
* **Phase output**: incremental $(\Delta u, \Delta v)$ for the camera chart.
* **Context**: morphological + non-morphological object descriptors, switch pulse, timing/metronome, optional intent bits.
* **Gating scope**: the launch build fixes $G_c=\mathbf{1}$ (no adaptive gating); the Examiner demo only exercises static priors and will graduate to learned gates once multi-column scenarios arrive.

### Explorer — Two Columns (Locomotion + Vision)

* **Column: Locomotion**
   * Features: proprioceptive bundle (wheel velocity, IMU pose deltas, contact sensors).
   * Phase input: planar ego-velocity $(v_{\text{fwd}}, v_{\text{turn}})$.
   * Phase output: $(\Delta s_{\text{fwd}}, \Delta s_{\text{turn}})$ motor command.
   * Context: shared global bits + compressed proprioceptive snapshot.
* **Column: Vision**
   * Features: SDR-encoded RGB-D patch (as in Examiner).
   * Phase input/output: pan/tilt velocities and commands.
   * Context: same global bits plus body-schema cues from locomotion.

Both columns fuse via shared entropy weighting; the locomotion column provides body-schema context to the vision column and vice versa through the context bus.

### Goalseeker (Future) — Manipulation + Reward

* **Arm Columns**: one per joint or effector.
   * Features: joint tactile/torque sensors, encoder deltas.
   * Phase input: $v_t = \dot q_c$ (joint velocity), output $\Delta q_c$ command.
   * Context: full body schema, goal/task embeddings, reward state bits.
* **Controller hint**: $\Delta q_c \propto J_c^{\top}\nabla_{\phi}U$ blending entropy reduction with extrinsic reward.

### Associative Maps (sparse, row-Top-K)

* $A_{FP}\in\mathbb{R}^{P\times D}$, $A_{PF}\in\mathbb{R}^{D\times P}$,
  $A_{PP}\in\mathbb{R}^{P\times P}$, $A_{FF}\in\mathbb{R}^{D\times D}$,
  $A_{CP}\in\mathbb{R}^{P\times C}$, $A_{CF}\in\mathbb{R}^{D\times C}$.

**Hebbian update (binary events) with decay**
$$
A_{ij} \leftarrow (1-\gamma)A_{ij} + \eta x_i y_j
$$
Apply **row-wise Top-(K)** pruning after each update.

---

## Path Integration (grid-module kinematics)

For module (m):
$$
\boldsymbol{\phi}_m \leftarrow (\boldsymbol{\phi}_m + R_m S_m \, \mathbf{v}_t \, \Delta t) \bmod 1
$$
$$
g_{\text{SDR}} = \bigoplus_{m=1}^M \text{TopK}_m\big(\text{bump}(\boldsymbol{\phi}_m)\big)
$$

**Pseudocode**

```
for m in modules:
    phi[m] = (phi[m] + R[m]*S[m]*v_t*dt) % 1.0
    g_m = topk_bump(phi[m], K_gm)
g_sdr = concat({g_m})
```

---

## Intra-Column Pipeline (with Context)

1. **Encode features**
   $$
   f^{(c)}_{\text{SDR}} = \mathrm{Encoder}^{(c)}(x^{(c)}_t)
   $$

2. **Motion prior (associative map projection)**
   $$
   g^{(c)}_{\text{prior}} = \mathrm{Assoc.phase\_from\_prior}\big(g^{(c)}_{\text{SDR,prev}}\big)
   $$

3. **Landmark evidence**
   $$
   g^{(c)}_{\text{feat}} = \mathrm{Assoc.phase\_from\_features}\big(f^{(c)}_{\text{SDR}}\big)
   $$

4. **Context prior with adaptive scaling**
   $$
   g^{(c)}_{\text{ctx}} = \mathrm{Assoc.phase\_from\_context}(G_c \odot c_{\text{SDR}}), \qquad
   \hat f^{(c)}_{\text{ctx}} = \mathrm{Assoc.feature\_from\_context}(G_c \odot c_{\text{SDR}})
   $$
   *Implementation note:* ship the first Examiner demo with $G_c=\mathbf{1}$ (static scaling). Adaptive gate updates remain a planned enhancement for multi-column experiments.

5. **Product-of-experts fusion**
   $$
   g^{(c)}_{\text{post}} = \alpha g^{(c)}_{\text{prior}} + \beta g^{(c)}_{\text{feat}} + \beta_c g^{(c)}_{\text{ctx}}
   $$
   $$
   g^{(c)}_{\text{SDR}} = \text{TopK}_g(\text{clip}(g^{(c)}_{\text{post}}))
   $$

6. **Completion (context-biased)**
   $$
   \hat f^{(c)} = \mathrm{Assoc.feature\_from\_phase}\big(g^{(c)}_{\text{SDR}}\big) + \mu_c \hat f^{(c)}_{\text{ctx}}
   $$
   $$
   \hat f^{(c)}_{\text{SDR}} = \text{TopK}_f(\hat f^{(c)})
   $$

7. **Optional blend**
   $$
   f^{(c)}_{\text{next}} = \text{TopK}_f\big(\lambda f^{(c)}_{\text{SDR}} + \mu \hat f^{(c)}_{\text{SDR}}\big)
   $$

**Pseudocode**

```
f_sdr = encoders.feature(column_id, x_t[column_id])
g_prior = assoc.phase_from_prior(prev_phase[column_id])
g_feat  = assoc.phase_from_features(f_sdr)
g_ctx   = assoc.phase_from_context(c_sdr["indices"])
g_post  = α*g_prior + β*g_feat + βc*g_ctx
g_sdr   = topk_g(clip(g_post))

f_ctx   = assoc.feature_from_context(scale_context(G_c, c_sdr["indices"]))
f_hat   = assoc.feature_from_phase(g_sdr) + μc*f_ctx
f_hat_sdr = topk_f(f_hat)

assoc.update(g_sdr, f_sdr, scaled_context_indices(G_c, c_sdr["indices"]))
```

---

## Multi-Column Consensus (PoE + entropy weights)

Local logits (per column) already incorporate prior, feature, and context evidence via the intra-column pipeline above. Consensus works in logits-space using entropy weights:
$$
w_c \propto \exp(-\tau H(g^{(c)}_{\text{post}})), \qquad g^*_{\text{logits}} = \sum_c w_c \, g^{(c)}_{\text{post}}.
$$
`ConsensusSystem.fuse(...)` computes these weights and returns $(g^*_{\text{logits}}, \{w_c\})$.
The shared SDR is
$$
g^*_{\text{SDR}} = \text{TopK}_g\big(\text{clip}(g^*_{\text{logits}})\big).
$$

Feedback and cross-column projections (e.g. $A_{PF}, A_{FF}$) are left as planned extensions once columns evolve beyond the current stubbed encoders.

---

## YCB-Habitat Environment (from scratch)

* **Agent chart:** 2D torus coordinates $(u,v)\in[0,1)^2$ per column.
* **Observation:** cropped RGB-D patch(s) at current $(u,v)$ on the object’s surface chart (or camera orbit with normalized spherical → chart mapping).
* **Context bits:** metronome (tick parity), object-switch pulse, optional intent bits from policy, room/task ID if applicable.
* **Actions:** `move(dx, dy)`, `switch_object`, `jump_to(u,v)`, `noop`.
* **Pose:** previous and current $(u,v)$ to derive $\mathbf{v}_t$.
* **Facet supervision:** (a) GT local surface from mesh render or depth; (b) predicted counterfactual facet from SDR completion; compare via 2.5D losses.

**Loop (single episode)**

```
load experiment config
env.reset(random_object=True, random_view=True)
for t in range(T):
    obs = env.render_patch(u,v)              # RGB-D patch; also returns pose packet & context bits
    x_t  = sensor.to_features(obs)           # e.g., downsampled grayscale vector or local descriptors
    v_t  = egomotion.pose_to_vt(pose_packet) # torus-shortest deltas
    c_t  = context_encoder(ctx_inputs)       # or pass-through bits from env/policy

    # SDR step (columns can be 1..C)
    belief = sdr.step(X_t={c: x_t_c}, V_t={c: v_t_c}, c_sdr=c_t)

    # Active sensing / random walk
    action = policy.act(belief, obs)

    # Optional: counterfactual facet synthesis at top-k shared phases
    cf = facets.predict(belief, obs.meta)

    env.step(action)

    # Logging & (if supervised) online updates for A_*
```

---

## 2.5D Facet Model & Counterfactual Reconstruction

**Representation:** Small fixed-res local height map or depth patch around a candidate allocentric phase cell (j):

* Facet $F(j) \in \mathbb{R}^{H_f\times W_f}$ (depth or signed distance), plus optional normal map $N(j)$.
* A learned/static **FeaturePrototypeBank** maps feature SDR indices $\hat f_{\text{SDR}}$ to primitive facets (or a linear decoder that “blits” prototypes).

**Counterfactual synthesis at shared hypothesis (j):**
$$
\hat f^{*}(j) = \sum_c w_c\big(A_{PF}^{(c)} e_j + \mu_c A_{CF}^{(c)} c\big)
\Rightarrow \hat F(j) = \text{decode\_facet}(\text{TopK}_f(\hat f^{*}(j)))
$$

**Comparison to rendered ground truth (same viewpoint or canonicalized):**

* **Depth loss:** $\mathcal{L}_D = \lvert \hat F - F_{\text{GT}} \rvert_1$ on valid mask
* **Normal loss (optional):** $\mathcal{L}_N = 1 - \langle \hat N, N_{\text{GT}} \rangle$
* **Chamfer (optional):** between back-projected point sets.
* **Peakiness/entropy gating:** only evaluate where $g^*$ is sufficiently peaky.

---

## Learning

**Per column (binary events):**
$$
\begin{aligned}
A_{FP}&\leftarrow(1-\gamma_{fp})A_{FP}+\eta_{fp} f g^\top\\
A_{PF}&\leftarrow(1-\gamma_{pf})A_{PF}+\eta_{pf} g f^\top\\
A_{PP}&\leftarrow(1-\gamma_{pp})A_{PP}+\eta_{pp} g_{\text{prev}} g^\top\\
A_{FF}&\leftarrow(1-\gamma_{ff})A_{FF}+\eta_{ff} f_{\text{prev}} f^\top\\
A_{CP}&\leftarrow(1-\gamma_{cp})A_{CP}+\eta_{cp} c g^\top\\
A_{CF}&\leftarrow(1-\gamma_{cf})A_{CF}+\eta_{cf} c f^\top
\end{aligned}
$$
Row-Top-(K) prune each map after update.

---

## BeliefPacket (motor-facing contract)

* `g_star_logits: float[P]`, `g_star_sdr: int[K_g*]`
* `per_column[c]: { g_post_logits: float[P_c], g_sdr: int[K_g_c], f_sdr: int[K_f_c] }`
* `entropy: float`, `peakiness: float` (from shared logits)
* `c_sdr: {indices: [..], length: C}`
* *(planned)* `helpers: { softmax, entropy, topk_g, topk_f }` convenience functions exposed alongside the packet (not yet in schema).

---

## Packets (contracts)

**ObservationPacket**

* `type:"observation.v1"`
* `columns:[{column_id, patch: HxWx{RGB,Depth}, egopose:{u,v,u_prev,v_prev}}]`
* `global_meta:{object_id, tick, camera_intr, …}`

**ContextPacket**

* `type:"context.v1"`
* `c_bits:{indices:[], length:C}`
* `sources:["metronome","switch","intent","heading","rate",…]`

**PosePacket**

* `type:"pose.v1"`
* `per_column:[{column_id, pose_t:{u,v}, pose_tm1:{u,v}}]`
* `dt: float`

**ActionMessage**

* `type: "move"|"switch_object"|"jump_to"|"noop"`
* `params:{dx,dy|u,v|…}`, optional `intent_bits`

**BeliefPacket**

* as above (validated by `belief.schema.json`)

**FacetRecord**

* `type:"facet.v1"`, `phase_idx:j`, `F_pred`, `F_gt`, `losses:{L1, normals, chamfer}`

**EvalRecord** *(contracts/eval.schema.json)*

* episode metrics: `{entropy_ts, peakiness_ts, facet_psnr, depth_L1, switch_acc, coverage}`

All contracts get JSON Schema with minimal required fields and enums.

---

## Simple Train/Eval Loop (single file)

**Pseudocode (current stub implementation; items in `[]` mark planned enrichments)**

```
cfg = load_config()
rng = np.random.default_rng(cfg.seed)
env = YCBEnv(cfg.env)
sensor = Sensor(cfg.sensor)
encoders = Encoders(cfg.encoder)
assoc = AssociativeMaps(cfg.assoc, rng)
consensus = ConsensusSystem(cfg.consensus)
facet_sys = FacetSynthesizer(cfg.facet, validator)
policy = RandomPolicy(cfg.policy, validator, rng)

state = env.reset()
prev_phase = {column_id: [] for column_id in env.columns}

for step in range(cfg.steps):
   obs, ctx, pose = env.observe()
   features = sensor.to_features(obs)
   context_sdr = encoders.context(ctx)
   G_c = context_gates.update(context_sdr, belief_entropy=None)
   # Examiner demo fallback: replace with G_c = ones_like(context_sdr) until adaptive gates ship

   column_logits = {}
   column_packets = {}

   for column_id, x_c in features.items():
      f_sdr = encoders.feature(column_id, x_c)
      g_prior = assoc.phase_from_prior(prev_phase[column_id])
      g_feat = assoc.phase_from_features(f_sdr)
      scaled_ctx = scale_context(G_c, context_sdr["indices"])
      g_ctx = assoc.phase_from_context(scaled_ctx)
      g_post = α*g_prior + β*g_feat + βc*g_ctx
      g_sdr = topk_g(g_post)

      assoc.update(g_sdr, f_sdr, scaled_ctx)
      prev_phase[column_id] = g_sdr

      column_logits[column_id] = g_post
      column_packets[column_id] = {
         "g_post_logits": handle("float32", (cfg.assoc.phase_dim,), storage_handle()),
         "g_sdr": make_sparse(g_sdr, cfg.assoc.phase_dim),
         "f_sdr": make_sparse(f_sdr, cfg.assoc.feature_dim),
      }

   g_star_logits, weights = consensus.fuse(column_logits)
   belief = make_belief_packet(g_star_logits, column_packets, context_sdr, weights)

   facet_records = facet_sys.predict(belief["g_star_sdr"]["indices"], obs)
   action = policy.act(belief)
   state = env.step(action)

   [log_eval(step, belief, facet_records)]
```

`examples/` will host an executable counterpart of this loop once the environment moves past the stub stage. For now `tests/test_loop_smoke.py` plus `main.py` provide the runnable contract.

`handle`, `make_sparse`, the placeholder `storage_handle()`, and the helper stubs `context_gates.update`, `scale_context`, `scaled_context_indices` align with utilities planned for `fusion.py`/`belief.py`.

The gate module keeps slice gains $G_c$ near unity; slices that reduce shared entropy receive boosts, while inactive slices decay. Optional RMS normalization inside `fusion.py` keeps logits numerically balanced before Top-K.

---

## Experiments

### Examiner — Single Column Baseline

* **Setup**: single YCB object on the orbit chart, one vision column, static context gates ($G_c=\mathbf{1}$). Action policy stays entropy-driven with occasional `switch_object` events to stress the contract.
* **What to measure**:
   * Entropy and peakiness trajectories while the column explores a fixed object.
   * Row-Top-K bounds on associative maps to ensure sparsity stays within 2–3%.
   * Counterfactual facet losses (depth L1 / PSNR) at the top shared phases.
   * Action counts (move vs switch vs jump) to verify entropy thresholds.
* **Expected behaviour**:
   * Entropy should decline over ~50–100 steps as the column re-observes familiar views.
   * Row-Top-K never exceeds configured limits; sparse maps converge.
   * Facet losses plateau at a low but non-zero value because the current decoder is synthetic.
   * Switches happen rarely (only when entropy exceeds the documented threshold).

### Explorer — Two Columns with Shared Context (future milestone)

* **Setup**: locomotion + vision columns operating in a small indoor layout with two objects. Shared context bits include head-direction and motion-rate; adaptive gating remains disabled until the Examiner stack is fully validated.
* **What to measure**:
   * Per-column vs shared entropy to confirm consensus improves certainty.
   * Cross-column influence: how often locomotion context bits modulate vision actions (e.g. dwell vs scan).
   * Safety metrics: collision counts, velocity clamps, and exploratory coverage of the chart.
   * Potential energy in the phase integrators (Top-K stability under continuous motion).
* **Expected behaviour**:
   * Shared entropy remains lower than either column individually once both streams are active.
   * Locomotion proposals avoid collisions and slow down when vision entropy is high (via context cues).
   * Coverage of the phase chart grows steadily; consensus weights oscillate but remain stable.
   * Logging shows coordinated ActionSet outputs with balanced weights.

### Goalseeker — Reward-Oriented Control (planned extension)

* **Setup**: multi-effector scene (e.g., 2–3 arm joints) with reward bits injected into the context SDR. Policies compute joint moves from Jacobian-transformed gradients. Environment emits sparse rewards when a target descriptor is matched.
* **What to measure**:
   * Blend of entropy reduction and reward accumulation across episodes.
   * Joint command magnitudes vs torque/velocity limits to ensure physical plausibility.
   * Goal progress signals (from context) and how they interact with adaptive gates once implemented.
   * Facet reconstructions in goal states to inspect whether desired objects are represented sharply.
* **Expected behaviour**:
   * Entropy does not necessarily converge to zero; reward peaks align with successful grasps/search.
   * Joint commands respect safety clamps; gradients remain bounded.
   * Context slices corresponding to reward states show elevated gains once adaptive gating is available.
   * Logs capture clear transitions between exploration (high entropy) and exploitation (high reward).

Each scenario is expressed as a compact experiment config (target <30 lines). The Examiner configuration doubles as the default example runner in `examples/`, while Explorer/Goalseeker configs stay dormant until their respective modules mature.

---

## Metrics

* **Entropy** $H$ of shared logits; **Peakiness** $\frac{H_{\max}-H}{H_{\max}}\in[0,1]$.
* **Localization:** phase index accuracy vs GT (synthetic).
* **Facet quality:** Depth L1, PSNR, (optional) normal cos, Chamfer.
* **Coverage:** fraction of phase cells visited/learned (bins hit).
* **Sparsity budgets:** active bits within 2–3% targets.

---

## Defaults (stable & safe)

* $K_g \approx 2\%$ per module, $K_f \approx 2\%$; $K_{CP},K_{CF}\in[8,32]$.
* Decays $\gamma_{\bullet}\in[5!\times!10^{-4},10^{-3}]$.
* Clip logits to $\pm 12$ before Top-K.
* Entropy temperature $\tau\in[1,3]$ for consensus weights.

---

## Tests (small, surgical)

* **`test_loop_smoke.py`** — 100 steps deterministically; packets validate; entropy finite.
* **`test_assoc_hebbian.py`** — toy co-activation strengthens correct rows; row-Top-K holds.
* **`test_consensus_ctx.py`** — context prior shifts ($g^*$) predictably; shared entropy reduces vs per-column.
* **`test_facet_recon.py`** — counterfactual facet PSNR improves after learning on a small suite of views.
* **`test_packets_schema.py`** — JSON validation for all packets; unknown fields rejected.
* **`test_pi_wraparound.py`** *(planned)* — torus shortest-delta correctness on edge cases.

---

## Why this design works

* **No nested config labyrinths** — Python *is* the wiring; packets are the contracts.
* **PoE everywhere** — clean additivity of priors/evidence; easy ablations.
* **Context as a bus** — a small, explicit bias path (CP/CF), not a new subsystem.
* **2.5D facets** — concrete counterfactuals you can render and compare; great for tests and demos.

