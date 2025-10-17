

---

# `ENVIRONMENT.md`

Minimal YCB-Habitat wrapper for SDR experiments.
Goals: deterministic packets, tiny API surface, and fast iteration. No external framework glue is assumed.

**Current status:** the codebase currently exposes a packet-generating stub inside `src/core/env_ycb.py`. Keep `EnvConfig.backend="stub"` (the default) to exercise the loop today and switch to `"habitat"` once the simulator adapter lands.

---

## 1) Scope & Scenarios

Supported scenarios (see `policy.md` for policies):

* **Examiner** — single YCB object; camera explores by pan/tilt on a 2D chart; may jump views or switch to a new object.
* **Explorer** — small world with multiple objects; agent navigates continuously (position/orientation) and optionally zooms.
* **Goalseeker (future)** — task/reward environment; same mechanics, adds extrinsic rewards and goal context.

This doc focuses on **Examiner** (MVP) and **Explorer** (continuous navigation). Goalseeker stubs are included only for interface completeness.

---

## 2) Dependencies & Assets

* **Habitat-Sim** (headless-capable) + minimal bindings.
* **YCB Object Models** (subset): meshes + textures stored under `assets/ycb/` (one directory per object id).
* **Examiner scene shell**: minimal black void GLB at `assets/scenes/examiner/void_black.glb`.
* **Explorer scenes**: Matterport3D GLBs rooted at `assets/scenes/mp3d/`. Default scene id: `17DRP5sb8fy/17DRP5sb8fy.glb`.

Version pinning recommended in your container or `requirements.txt`. Use asset indices as stable IDs.

---

## 3) Coordinate Frames & Charts

### 3.1 Examiner (2D chart / orbit)

* Internal control space is a **2D torus** $(u,v)\in[0,1)^2$.
* Orbit mapping from $(u,v)$ to camera pose around the object:

  * Azimuth $\theta(u)=2\pi u$, Elevation $\phi(v)=\phi_{\min}+(\phi_{\max}-\phi_{\min})v$.
  * Camera radius $r$ (fixed or slowly varying).
* “Chart” implies wrapping; deltas use **shortest torus difference**.

### 3.2 Explorer (allocentric SE(3))

* Agent state $s=(\mathbf{p}, \mathbf{R})$ with position $\mathbf{p}\in\mathbb{R}^3$ and yaw-pitch-roll (or quaternion) $\mathbf{R}$.
* Local motor vector $u\in[-1,1]^M$ mapped to $\Delta s = B,\tanh(u)$, then integrated to $s_{t+1}$.
* To feed the SDR **phase path integration**, reduce to a **2D allocentric velocity** per column, e.g., $(v_x, v_y)$ on a local chart (choice documented per experiment).

---

## 4) Observation → Feature Pipeline

Per step and per column:

1. **Render** a crop/patch (RGB, Depth, optional Normals/Segmentation) from Habitat at the current camera pose.
2. **Normalize** (e.g., grayscale + unit-range).
3. **Emit** `ObservationPacket`, `ContextPacket`, `PosePacket` (see `packets.md`).
4. **Sensor adapter** converts patch → feature tensor $x_t$ (kept thin; SDR encoding happens in `encoders.py`).

**Patch sampling:** For orbits, crop the object within a tight FOV; for Explorer, use first-person camera with fixed intrinsics (recorded in `global_meta.camera_intr`).

---

## 5) Actions & Column Interfaces

### 5.1 Action Types (unified)

* `move(dx, dy)` — continuous chart motion (Examiner) **or** continuous SE(3) control (Explorer) mapped internally.
* `jump_to(u, v)` — discrete reposition on chart (Examiner).
* `switch_object` — discrete object swap.
* `noop`
* Optional (Explorer): `rotate(dyaw, dpitch)`, `zoom(dz)`, `focus(dƒ)`.

### 5.2 Column-Centric Phase Updates

Each column owns a 2D allocentric phase and a matched 2D motor command. Policies read the per-column pose deltas from `PosePacket`, integrate them locally (see `phase.py`), and emit the next 2D motor increment.

---

## 6) Context Signals

`ContextPacket.c_bits` is a sparse vector whose exact sources depend on the active scenario:

* **Examiner** — global morphological descriptors (e.g., object category hash, symmetry flags), non-morphological context (lighting/scene tags), and the `switch_pulse` bit that fires on object changes.
* **Explorer** — extends Examiner bits with **head direction bins**, **motion rate** (fast/slow), **intent** toggles (“explore vs examine”), and a multi-rate **metronome** clock to coordinate multi-column behaviour.
* **Goalseeker** *(planned)* — includes all Explorer signals plus a compact **reward state** encoding (recent reward sign/magnitude, goal progress buckets).

All indices are stable and documented; `packets.md` defines a canonical length `C` and source index ranges.

---

## 7) Episode & Step Semantics

### 7.1 Reset

```python
obs, ctx, pose = env.reset(
    object_id=None,            # None → random choice
    random_view=True,          # Examiner: random (u,v); Explorer: random pose
    seed=None,                 # optional RNG seed for determinism
    scenario="examiner"        # or "explorer"
)
```

### 7.2 Step

```python
obs, ctx, pose, info = env.step(action)
```

* Applies action to internal state (wrap on chart; integrate in SE(3) for Explorer).
* Renders observations; updates context and pose packets.
* Returns lightweight `info` (e.g., collisions, clamp/wrap flags, object_id).

**Timing:** `PosePacket.dt` is constant (configurable). Path integration uses torus-shortest delta:

[
\Delta u = \mathrm{wrap}(u_t - u_{t-1}),\quad
\Delta v = \mathrm{wrap}(v_t - v_{t-1}),\quad
\mathbf{v}_t = \frac{1}{\mathrm{dt}},[\Delta u,\ \Delta v]
]

with `wrap(d) = ((d + 0.5) \bmod 1.0) - 0.5`.

---

## 8) Examiner Details (MVP)

**State:** single object index, chart coords $(u,v)$.

**Actions:**

* `move(dx, dy)`:
  $(u,v)\leftarrow (u + s,dx,\ v + s,dy) \bmod 1$, with step scale `s`.
* `jump_to(u, v)`:
  set directly, no velocity carry-over.
* `switch_object`:
  cycle or sample a new object; set `switch_pulse=1` in `ContextPacket`.

**Rendering:** small RGB-D patch; optional normals from G-buffer.

**Environment:** default scene is the `void_black.glb` shell positioned at the origin with a single YCB object resting on an invisible table plane. Lighting is neutral to emphasise object albedo.

**Evaluation hooks:** facet depth/normal ground-truth extracted at the same pose for reconstruction loss/eval.

---

## 9) Explorer Details (continuous navigation)

**State:** $s=(\mathbf{p},\mathbf{R})$ with collision-safe integration.

**Actions:**

* `move(dx, dy)`: planar velocity in agent frame (map to $x,y$ translation).
* Optional extra DOFs:

  * `dz` (zoom or dolly),
  * `dyaw, dpitch` (view orientation),
  * actuator controls if you later add manipulation.

**Safety:** physics/collision queries enforce step feasibility; clamp with informative `info` flags (used by policy cost).

**Environment:** defaults to the Matterport3D home layout `17DRP5sb8fy` located at `assets/scenes/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb`. Replace the GLB path in `AssetConfig.explorer_scene` to target alternative scans.

**Chart coupling:** if your SDR columns expect 2D phases, define a **chart-of-interest** (e.g., object-centric $u,v$ inferred from current gaze ray) and expose it in `PosePacket` alongside full SE(3).

---

## 10) Facet Supervision

Two sources:

1. **Depth reprojection:** from the current render buffer (Examiner & Explorer).
2. **Mesh sampling:** sample local surface patch around gaze intersection for canonical facet comparison.

Use whichever matches your `facet.py` decoder’s coordinate frame; record which variant was used in `FacetRecord`.

---

## 11) Configuration Schema

`EnvConfig` encapsulates all environment-facing parameters. It is a dataclass composed of four nested helpers and is shared by the stub adapter and the future Habitat-backed implementation. The parser lives in `src/core/env_ycb.py` (`EnvConfig.from_dict(...)`) and enforces the following schema:

* **Top-level (`EnvConfig`)**
  * `scenario`: `"examiner"` or `"explorer"`. Controls orbit vs navigation semantics.
  * `backend`: `"stub"` or `"habitat"`. Selects between the packet stub and real simulator.
  * `columns`: tuple of column identifiers (default `("col0",)`).
  * `context_length`: positive integer; drives context encoder shape (default `1024`).
  * `dt`: fixed step duration in seconds (default `0.05`).
  * `objects`: tuple of YCB object ids (default `("003_cracker_box", "005_tomato_soup_can", "006_mustard_bottle")`).
  * `patch_shape`: optional `(H, W, C)` override. If omitted, derived from `sensor.resolution + sensor.modalities`.
* **`SensorConfig`**
  * `resolution`: `(height, width)` tuple; defaults to `(64, 64)` for the stub.
  * `modalities`: tuple of strings (e.g. `("rgb", "depth")`). Controls channel count and Habitat sensor attachments.
  * `hfov`: horizontal field of view in degrees (default `70.0`).
  * `near` / `far`: clipping planes in metres; require `0 < near < far`.
* **`OrbitConfig`** (Examiner)
  * `radius`: camera-object distance in metres (default `0.6`).
  * `min_elevation` / `max_elevation`: allowable vertical angles (defaults `-0.55`, `0.65` radians).
  * `jitter`: stochastic perturbation applied on reset (default `0.0`).
  * `default_speed`: chart velocity scale used when policies emit unit actions (default `0.05`).
* **`PhysicsConfig`**
  * `enable_physics`: toggles Habitat physics engine (default `False` for Examiner).
  * `enable_sliding`: when physics is active, allow slide resolution instead of hard stops (default `True`).
  * `lock_object`: keep the target object frozen in place (default `True`).
* **`AssetConfig`**
  * `ycb_root`: filesystem path containing YCB object assets (default `assets/ycb`).
  * `scene_root`: root directory for static scenes (default `assets/scenes`).
  * `examiner_scene`: GLB relative to `scene_root` used for Examiner (`examiner/void_black.glb`).
  * `explorer_scene`: GLB relative to `scene_root` for Explorer (`mp3d/17DRP5sb8fy/17DRP5sb8fy.glb`).

`EnvConfig.from_dict` accepts plain dictionaries (JSON/YAML payloads) and returns a validated config instance. All numeric values are coerced to floats/ints, sequences are converted to tuples, and invalid combinations raise `ValueError` with descriptive messages.

---

## 12) Performance & Headless

* **Headless first**: EGL/OSMesa. Provide `--headless` switch.
* Target **>120 FPS** for small patches on a mid-range GPU in Examiner.
* Enable **frame skipping** or **lower resolution** for CI.
* Batch simulation is optional; MVP is single instance, single GPU.

---

## 13) Testing & Tooling

Current automated coverage (`tests/`):

* `test_loop_smoke.py`: deterministic 100-step loop exercising reset/step and packet emission.
* `test_packets_schema.py`: validates Observation/Context/Pose/Action/Belief/Facet/Eval packets against JSON Schema.
* `test_consensus_ctx.py`: sanity-checks context priors flowing through consensus logits.
* `test_assoc_hebbian.py`: verifies Hebbian updates respect row-Top-K constraints.
* `test_facet_recon.py`: exercises facet prediction + loss wiring on toy data.

Planned additions:

* `test_scene_load`: asset resolution + baseline render on CI assets.
* `test_torus_wrap`: explicit wraparound edge cases for pose deltas.
* `test_column_phase_update`: ensure per-column phase integrator matches torus deltas.
* `test_facet_gt`: mesh-derived facet ground truth parity check.

**Debug tools:**

* `env.debug_overlay=True` draws pose axes, object bbox, and patch frustum on a low-res preview.
* `dump_step(n)` writes PNG patches + JSON packets (handy for repro).

---

## 14) Failure Handling

* **Missing assets** → raise `EnvAssetError(object_id, path)`.
* **Physics explosions / NaNs** → rollback to last safe state; set `info.reset_suggested=True`.
* **Renderer errors** → retry once; on fail, mark step invalid and return `noop` effects with flags.
* **Contract violations** → assert early via schema checks in debug mode.

---

## 15) Reference Step (Examiner) — Pseudocode

```python
def reset(cfg):
    self.obj = choose(cfg.objects)
    self.u, self.v = rng.random(), rng.random()
    self.tick = 0
    return observe()

def step(action):
    self.tick += 1
    # --- apply action
    if action.type == "move":
        s = cfg.step_size
        self.u = (self.u + s * action.params["dx"]) % 1.0
        self.v = (self.v + s * action.params["dy"]) % 1.0
    elif action.type == "jump_to":
        self.u, self.v = action.params["u"] % 1.0, action.params["v"] % 1.0
    elif action.type == "switch_object":
        self.obj = choose(cfg.objects)
        self.switch_pulse = 1
    elif action.type == "noop":
        pass
    else:
        raise ValueError("unknown action")

    return observe()

def observe():
    # prev/current for PosePacket
    pose = {
      "type": "pose.v1",
      "per_column": [{
          "column_id": "col0",
          "pose_t":   {"u": self.u, "v": self.v},
          "pose_tm1": {"u": self.u_prev, "v": self.v_prev}
      }],
      "dt": cfg.dt
    }
    # render patch
    patch = habitat_render(self.obj, self.u, self.v, cfg)
    observation = {
      "type": "observation.v1",
      "columns": [{
          "column_id": "col0",
          "patch": patch,
          "egocentric_pose": {"u": self.u, "v": self.v,
                              "u_prev": self.u_prev, "v_prev": self.v_prev}
      }],
      "global_meta": {"object_id": self.obj, "tick": self.tick,
                      "camera_intr": cfg.camera_intr}
    }
    # context bits
    indices = []
    if self.tick % 2 == 0: indices.append(0)         # metronome
    if self.switch_pulse:  indices.append(1)         # switch
    ctx = {"type":"context.v1", "c_bits": {"indices": indices, "length": C},
           "sources": ["metronome","switch","intent","heading","rate"]}

    self.u_prev, self.v_prev = self.u, self.v
    self.switch_pulse = 0

    return observation, ctx, pose, {"ok": True}
```

---

## 16) Policy/Utility Hook Points

* `phase.shift(column_id, pose_packet)`: updates a column’s local phase state from the latest torus deltas.
* `project_motor(column_id, action)`: translates JSON actions back into simulator control vectors (per column).
* `safety(info)`: inspects clamp/collision flags returned by the step for policy-side cost shaping.

The environment stays utility-agnostic; it surfaces accurate packets and per-column kinematics and leaves scoring/selection to the policy.

---

## 17) What’s deliberately missing

* No baked-in reward shaping (except placeholder in Goalseeker).
* No heavyweight dataset generators; recordings are produced by the main loop via `EvalRecord`.
* No monolithic config systems — Python objects are the config.

---

