---

# `POLICY.md`

## Column-Centric Action Model

Policies consume the `BeliefPacket` and operate **per column**. Each column maintains its own 2D motor chart, utility field, and entropy estimate; shared behaviour emerges by weighting the columns with entropy-derived gains and the adaptive context gates described in this design blueprint.

Key ingredients shared across scenarios:

* `utility_field[c]`: phase-aligned 2D grid storing post-fusion utility estimates for column $c$.
* `H_c`: column entropy computed from `g_post_logits`.
* `w_c = \mathrm{softmax}(-\tau H_c)`: consensus-style weight used to blend column proposals.
* `context_gates.update(...) → G_c`: adaptive scale factors reused by policy-side heuristics (e.g., down-weight locomotion when intent bits favour inspection).

The primitive action API remains the JSON contract in `contracts/action.schema.json`. Policies may return a singular `Action` or an `ActionSet` bucketed by column ID.

For any candidate action $a$ proposed by column $c$ we evaluate

$$
U_c(a) = \left[H_c - H_c(a)\right] + \lambda_{\text{peak}}\,\mathrm{peak}(a) - \lambda_{\text{cost}}\,\mathrm{cost}(a),
$$

where $H_c(a)$ is the predicted entropy after applying $a$, `peak` measures the sharpness of the resulting logits (max or mean of Top-K), and `cost` captures motor expenditure or safety penalties. Scenario-specific sections detail how $a$ is sampled and scored.

---

## 1. Examiner — *Curiosity on a Single Object*

**Columns:** one vision column controlling camera pan/tilt.

**Inputs:** feature SDR from RGB-D patch, phase prior, context bits covering morphology, switch pulse, metronome.

**Action space:**

* Continuous `move(dx, dy)` within the torus chart.
* Discrete `switch_object`, `jump_to(u, v)`, or `noop`.

**Selection logic:**

* Trigger `switch_object` when shared entropy $H^*$ exceeds $H_{\text{switch}}$ for $N$ successive ticks.
* Otherwise, sample $(\Delta u, \Delta v)$ from the column utility field via soft-argmax or SPSA; clamp to chart limits and emit `Action("move", ...)`.
* If entropy reduction stalls (slope $<\epsilon$ for $M$ steps) choose a stochastic `jump_to` seeded by the top counterfactual peaks.

`context_gates` down-weight intent cues when the examiner should dwell (e.g., `intent.inspect`). Gains also modulate the discrete switch threshold so high-confidence contexts resist premature switching.

---

## 2. Explorer — *Two Columns, Shared Curiosity*

**Columns:**

1. **Locomotion** (planar motion): proposes $(\Delta s_{\text{fwd}}, \Delta s_{\text{turn}})$.
2. **Vision** (pan/tilt): proposes $(\Delta u, \Delta v)$.

**Context interplay:** locomotion context bits (wheel slip, contact pulse, cell coverage) are projected through `A_{CF}` and re-used as inputs for the vision column’s gate; vision saliency bits reciprocally bias locomotion.

**Action space:** both columns emit continuous commands; final step produces an `ActionSet` aggregating their proposals.

**Selection logic:**

1. Each column generates a shortlist of candidates via gradient-ascent over its `utility_field` (1–3 SPSA steps) and a safety-filter that respects velocity limits.
2. Evaluate $U_c(a)$ per candidate using updated entropy predictions.
3. Combine winners with entropy weights $w_c$ and optional gate multipliers (e.g., suppress locomotion when `metronome` signals a dwell phase).
4. Emit `ActionSet({"locomotion": ..., "vision": ...})` where each payload is already clamped to hardware bounds.

Each column integrates directly with `phase.shift` utilities inside `phase.py`.

---

## 3. Goalseeker — *Reward-Directed Control* (Planned)

**Columns:** one per manipulator joint or effector; may extend to symbolic planners.

**Additional inputs:** goal SDR slice, reward traces, task embeddings.

**Action space:** continuous `joint_move` actions, optional discrete `switch_goal` or `commit_plan` events.

**Selection logic:**

* Form a blended objective $U = \Delta H + w_r r_t + w_g\,\text{align}(c_{\text{goal}})$ where weights are column-specific gate outputs.
* Each column computes $\Delta q_c = \eta_c J_c^{\top} \nabla_{\phi} U$ using Jacobians from the manipulator model; safety guards enforce torque limits.
* Assemble `ActionSet({column_id: Action("joint_move", dq=\Delta q_c)})`.

Implementation of reward-aware utilities, Jacobians, and goal SDR encoders remains future work; stubs live in `policy.py` and `context.py`.

---

## Unified Policy Skeleton

```python
def policy_act(belief: BeliefPacket, obs, scenario: str):
	gates = context_gates.update(belief.context_bits, belief.entropy)
	weights = entropy_weights(belief.per_column)

	if scenario == "examiner":
		if belief.entropy > thresholds.switch(gates):
			return Action("switch_object")
		du, dv = sample_continuous(belief.utility_field["vision"], gates)
		if stalled_entropy(belief.history):
			return Action("jump_to", u=du, v=dv)
		return Action("move", dx=du, dy=dv)

	if scenario == "explorer":
		loco_cmd = optimise_column("locomotion", belief, gates, weights)
		vision_cmd = optimise_column("vision", belief, gates, weights)
		return ActionSet({"locomotion": loco_cmd, "vision": vision_cmd})

	if scenario == "goalseeker":
		grads = compute_goal_gradients(belief, gates)
		return ActionSet({c: Action("joint_move", dq=grads[c]) for c in grads})

	return Action("noop")
```

Helper hooks such as `entropy_weights`, `stalled_entropy`, `sample_continuous`, and `optimise_column` are planned utilities to be implemented alongside the adaptive gating helpers in `fusion.py` / `policy.py`.

---

## Integration Points

* **BeliefPacket:** must expose per-column entropy, utility fields, and context indices; current stub only surfaces entropy and logits, so utility projections are pending.
* **RandomPolicy (current code):** remains the baseline stub, ensuring contract compliance while the structured policies are prototyped.
* **Loop wiring:** `loop.step` will call `policy_act`, validate the returned `Action`/`ActionSet`, and pass through to `env.step`.
* **Tests:** planned additions (`test_policy_examiner.py`, `test_policy_explorer.py`) will validate gating behaviour, entropy thresholds, and schema compliance.

---

## Scenario Summary

| Policy       | Columns           | Output                    | Optimisation    | Primary Objective                  |
| ------------ | ----------------- | ------------------------- | ----------------| ---------------------------------- |
| Examiner     | Vision            | Single `Action`           | Soft-argmax / SPSA | $\Delta$Entropy with dwell control |
| Explorer     | Locomotion + Vision | `ActionSet` per column  | SPSA + entropy weights | $\Delta$Entropy − cost (safety)    |
| Goalseeker   | Arms / effectors  | `ActionSet` of joint commands | Gradient via $J^T \nabla U$ | $\Delta$Entropy + reward + goal   |

All scenarios share the entropy-centric utility but differ in column wiring, context gates, and choice of primitive actions. Every column acts in its own 2D control surface aligned with the SDR phase charts.

