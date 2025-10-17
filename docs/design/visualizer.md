# `VISUALIZER.md`

## Goals

Design a mixin-style visualisation layer that can be attached to the main Examiner loop to render three synchronous views in real time:

1. The current object frame from the YCB environment (full RGB render).
2. A tiled panel showing the observation patches each column receives.
3. Live counterfactual 2.5D facets corresponding to the top shared phase hypotheses.

The mixin should be lightweight, optional, and integrate cleanly with the existing loop without breaking headless execution.

## Architectural Plan

### 1. Integration Points

- **Loop hook:** extend `ColumnSystem.step` or `run_loop` to emit a `VisualizerPayload` containing observation patches, belief state metadata, and facet tensors after each iteration.
- **Packet sources:** reuse the already validated ObservationPacket, PosePacket, and FacetRecord outputs; avoid duplicating data.
- **Mixin attachment:** the loop instantiates a `Visualizer` if a config flag is true; the visualizer registers callbacks for `on_step_start`, `on_step_end` to process payloads.

### 2. Visual Streams

- **Environment view:** leverage the Habitat adapter to expose the RGB frame before downsampling. Render via matplotlib or OpenCV in a non-blocking window; fall back to saving frames when headless.
- **Column patch grid:** stack per-column patches (e.g., `col0` top-left, `col1` top-right) with annotations showing entropy and current action.
- **Facet reconstructions:** decode the shared facet handles into numpy arrays (stub currently uses random handles; future version will fetch actual tensors). Display the top-K facets with PSNR/L1 overlay.

### 3. Real-time Considerations

- Use a dedicated thread or async loop to avoid blocking the main SDR computation.
- Maintain a rolling buffer (e.g., last 10 frames) so GUI lag does not stall the policy.
- Provide graceful degradation: if GUI libraries are unavailable, log a warning and disable visualisation.

## Implementation Steps

1. **Define Visualizer Protocol**
   - Create `VisualizerMixin` class with methods `begin_episode`, `render_step(payload)`, `end_episode`.
   - The mixin accepts config for window titles, update frequency, and headless mode.

2. **Add Payload Struct**
   - Introduce dataclass `VisualizerPayload` with fields: `rgb_frame`, `column_patches`, `column_actions`, `facet_arrays`, `metrics`.
   - Construct this payload in the loop after facet synthesis.

3. **Rendering Backend**
   - Implement a default backend using matplotlib (`plt.imshow`) in interactive mode (`plt.ion()`), updating subplots for each stream.
   - Optional: design an OpenCV backend for faster updates when available.

4. **Headless / Recording Mode**
   - Support a record-to-disk path: save composite frames as PNG/MP4 using imageio when `config.visualizer.record=True`.
   - Allow switching to pure logging when no display is attached.

5. **Configuration Wiring**
   - Extend `LoopConfig` with `visualizer: Optional[VisualizerConfig]`.
   - The example runner reads `--viz` flag to enable the mixin.

6. **Testing Strategy**
   - Provide a unit test that instantiates the visualizer in headless mode, feeds synthetic payloads, and ensures frame buffers update.
   - Add integration test running a short loop with `record=True` and verify output directory contains expected PNG/JSON metadata.

## Open Questions

- Choice of backend (matplotlib vs. vispy) for real-time updates; start with matplotlib for simplicity.
- Efficient retrieval of facet tensors once synthetic handles are replaced with real data.
- Synchronisation with Habitat camera pose should we add a real renderer in the future.

## Next Actions

1. Prototype `VisualizerMixin` with matplotlib backend.
2. Wire mixin into `run_loop` with config toggle.
3. Document usage in `README.md` and the Examiner runner example.
