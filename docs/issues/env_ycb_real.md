# Task: Replace stub YCB environment with real Habitat-backed adapter

## Overview

The current `YCBHabitatAdapter` in `src/core/env_ycb.py` fabricates packets but does not invoke Habitat-Sim. To progress past the Examiner demo, we must implement a real environment wrapper that streams RGB-D observations, pose, and context bits from Habitat assets. This issue tracks the migration path and required subtasks.

## Plan of Attack

1. **Establish Habitat dependency baseline**
   - Confirm `environment.yml` pins suitable Habitat-Sim / Habitat-Lab versions.
   - Add sanity check script to verify imports and GPU availability (optional fallback to CPU).

2. **Define configuration schema**
   - Expand `EnvConfig` to include asset paths (`ycb_root`, `scene_root`), sensor intrinsics, orbit radius, physics toggles.
   - Document defaults and required fields in `docs/environment.md`.

3. **Implement Habitat loader**
   - Create helper to initialise Habitat-Sim simulator with target scene and sensors.
   - Support random object sampling and initial camera placement matching the Examiner chart.

4. **Observation pipeline**
   - Render RGB-D frame per step, convert to numpy arrays, and package into `ObservationPacket` using shared-memory handles or direct arrays.
   - Add conversion for patch crops and ensure they obey existing packet schemas.

5. **Pose & context updates**
   - Derive torus coordinates `(u,v)` from camera pose relative to the object; maintain previous pose for velocity deltas.
   - Emit context bits for metronome, switch pulses, intent—as currently used by the loop.

6. **Action application**
   - Map `ActionMessage` (`move`, `jump_to`, `switch_object`) to Habitat API calls that adjust camera position/orbit.
   - Handle collision / clamp enforcement and annotate `info` dictionary with status flags.

7. **Testing & validation**
   - Write integration test that spins up the simulator for a handful of steps to ensure packet schemas validate.
   - Update `test_loop_smoke.py` (future variant) to run against the real adapter when assets are available (skip when missing).

8. **Documentation updates**
   - Revise `docs/environment.md` with setup instructions, asset requirements, and performance notes.
   - Extend `README.md` and `examples/examiner_minimal.py` to mention the real environment toggle.

9. **Fallback / feature flags**
   - Keep stub adapter as fallback (`use_stub=True`) so CI and rapid tests can stay lightweight.
   - Provide CLI flag or config field to switch between stub and Habitat backend.

## Deliverables

- Habitat-backed implementation in `src/core/env_ycb.py` (or new module) with unit/integration tests.
- Updated documentation describing environment setup.
- Example/Examiner runner capable of toggling real vs stub environments.
