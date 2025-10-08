Below is the updated spec with environment details (YCB Habitat), equations, pseudocode, pipeline, experiments, contracts, and tests — nothing else.

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
    env_ycb.py          # thin YCB Habitat wrapper → Observation/Context/Pose
    sensor.py           # patch → feature vector
    phase.py            # grid modules + path integration (PI)
    encoders.py         # HebbianSPEncoder / FeaturePrototypeBank / ContextEncoder
    assoc.py            # A_FP, A_PF, A_PP, A_FF, A_CP, A_CF (row-Top-K)
    fusion.py           # PoE fusion: prior + features + context
    consensus.py        # multi-column PoE over shared phase
    belief.py           # BeliefPacket + helpers (entropy, peakiness, top-k)
    facet.py            # 2.5D facet representation + renderer + losses
    policy.py           # RandomWalk + ActiveSensing (context-aware)
    loop.py             # train/eval loop (single file, <~200–300 LOC)
  contracts/
    observation.schema.json
    context.schema.json
    pose.schema.json
    action.schema.json
    belief.schema.json
    facet.schema.json
    eval_record.schema.json
  examples/
    ycb_minimal.py      # end-to-end demo (runs in <1 min)
  tests/
    test_loop_smoke.py
    test_assoc_hebbian.py
    test_consensus_ctx.py
    test_facet_recon.py
    test_packets_schema.py
  docs/
    DESIGN.md           # this blueprint
    PACKETS.md          # packet tables + JSON snippets
    STYLE.md            # coding/style paradigm
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
* Encodes metronome/tempo, intent, room/task, head direction, rate control, etc.

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

2. **Motion prior**
   $$
   g^{(c)}_{\text{prior}} = A^{(c)}_{PP} g^{(c)}_{\text{SDR,prev}}
   $$

3. **Landmark evidence**
   $$
   g^{(c)}_{\text{feat}} = A^{(c)}_{FP} f^{(c)}_{\text{SDR}}
   $$

4. **Context prior**
   $$
   g^{(c)}_{\text{ctx}} = A^{(c)}_{CP} c_{\text{SDR}}, \qquad
   \hat f^{(c)}_{\text{ctx}} = A^{(c)}_{CF} c_{\text{SDR}}
   $$

5. **Product-of-experts fusion**
   $$
   g^{(c)}_{\text{post}} = \alpha g^{(c)}_{\text{prior}} + \beta g^{(c)}_{\text{feat}} + \beta_c g^{(c)}_{\text{ctx}}
   $$
   $$
   g^{(c)}_{\text{SDR}} = \text{TopK}_g(\text{clip}(g^{(c)}_{\text{post}}))
   $$

6. **Completion (context-biased)**
   $$
   \hat f^{(c)} = A^{(c)}_{PF} g^{(c)}_{\text{SDR}} + \mu_c \hat f^{(c)}_{\text{ctx}}
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
f[c] = encoder[c](x_t[c])
g_prior[c] = A_PP[c] @ g_prev[c]
g_feat[c]  = A_FP[c] @ f[c]
g_ctx[c]   = A_CP[c] @ c_sdr
g_post[c]  = α*g_prior[c] + β*g_feat[c] + βc*g_ctx[c]
g[c]       = topk_g(clip(g_post[c]))

f_ctx[c]   = A_CF[c] @ c_sdr
f_hat[c]   = A_PF[c] @ g[c] + μc*f_ctx[c]
f_hat_sdr[c] = topk_f(f_hat[c])
```

---

## Multi-Column Consensus (PoE + entropy weights)

Local logits (already include context):
$$
   ilde g^{(c)} = \alpha A_{PP}^{(c)} g_{\text{prev}}^{(c)} + \beta A_{FP}^{(c)} f^{(c)} + \beta_c A_{CP}^{(c)} c
$$

Shared product-of-experts:
$$
g^* = \gamma g_{\text{PI}} + \sum_c w_c \, \Pi^{(c\to *)}(\tilde g^{(c)}) + \gamma_c \bar A_{CP} c
$$
$$
w_c \propto \exp\big(-\tau H(\tilde g^{(c)})\big), \quad g^*_{\text{SDR}} = \text{TopK}_g(\text{clip}(g^*))
$$

Feedback:
$$
g^{(c)} \leftarrow \text{TopK}_g\big(\rho \, \Pi^{(c\to *)}(g^{(c)}) + \kappa \, \Pi^{(*\to c)}(g^*)\big)
$$
$$
f^{(c)} \leftarrow \text{TopK}_f\big(\lambda f^{(c)} + \mu (A_{PF}^{(c)} g^{(c)}) + \nu (A_{FF}^{(c)} f^{(c)}) + \mu_c A_{CF}^{(c)} c\big)
$$

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
* `helpers: { softmax, entropy, topk_g, topk_f }`

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

**EvalRecord**

* episode metrics: `{entropy_ts, peakiness_ts, facet_psnr, depth_L1, switch_acc, coverage}`

All contracts get JSON Schema with minimal required fields and enums.

---

## Simple Train/Eval Loop (single file)

**Pseudocode**

```
cfg = load_config()
env = YCBEnv(cfg.env)              # random object, random view
sensor = Sensor(cfg.sensor)
enc = Encoders(cfg.encoder)        # feature + (optional) context
phase = PhaseModules(cfg.phase)
assoc = AssocMaps(cfg.assoc)
cons = Consensus(cfg.consensus)
policy = make_policy(cfg.policy)
facets = FacetSystem(cfg.facet)

state = env.reset()
for t in range(cfg.steps):
    obs, ctx, pose = env.observe()
    X_t = sensor.to_features(obs)
    V_t = phase.pose_to_vt(pose)   # PI velocity
    c_t = enc.context(ctx)

    # Intra-column + consensus
    per_col = []
    for c in columns:
        f_sdr = enc.feature[c](X_t[c])
        g_prior = assoc.A_PP[c] @ phase.g_prev[c]
        g_feat  = assoc.A_FP[c] @ f_sdr
        g_ctx   = assoc.A_CP[c] @ c_t
        g_post  = α*g_prior + β*g_feat + βc*g_ctx
        g_sdr   = topk_g(clip(g_post))
        if train: assoc.update_all(c, f_sdr, g_sdr, c_t)  # FP, PF, PP, FF, CP, CF
        per_col.append((g_post, g_sdr, f_sdr))

    g_star = cons.product_of_experts(per_col, c_t)
    belief = make_belief(g_star, per_col, c_t)

    # Counterfactual facet at top shared phases
    cf = facets.predict(belief, obs)

    action = policy.act(belief, obs)
    env.step(action)
```

---

## Experiments

**E0 — Smoke (single column, FP only)**

* Goal: loop stability, sparsity sanity.
* Metric: entropy↓ over time on static view; row-Top-K bounded.

**E1 — Motion prior (PP)**

* Add PI + (A_{PP}).
* Metric: localization error on synthetic trajectory, entropy vs speed.

**E2 — Multi-column consensus**

* Two columns (different view crops) of same object.
* Metric: shared entropy < per-column average; top-k phase stability.

**E3 — Context priors (CP/CF)**

* Toggle metronome / object-switch bits.
* Metric: ablation Δentropy with/without context.

**E4 — 2.5D facet reconstruction**

* Predict facet at top-k phases, compare to depth/mesh render.
* Metrics: depth L1 / PSNR, optional Chamfer, normal cosine.

**E5 — Object switches**

* Random switch events; verify loop-closure behaviour & gating.

Each experiment is a <30-line config and a call to `examples/ycb_minimal.py`.

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
* **`test_pi_wraparound.py`** — torus shortest-delta correctness on edge cases.

---

## Why this design works

* **No nested config labyrinths** — Python *is* the wiring; packets are the contracts.
* **PoE everywhere** — clean additivity of priors/evidence; easy ablations.
* **Context as a bus** — a small, explicit bias path (CP/CF), not a new subsystem.
* **2.5D facets** — concrete counterfactuals you can render and compare; great for tests and demos.

