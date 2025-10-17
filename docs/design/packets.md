# PACKETS.md — Minimal Wire Contracts for the SDR Loop

This file defines the compact, stable message shapes the end-to-end loop passes around. Each packet has:

* a terse JSON example (copy-pasteable),
* a **draft-07 JSON Schema stub** (kept small on purpose),
* brief field notes where ambiguity could bite.

All IDs/indices are zero-based. Floats are `number`. Sparsity uses `{indices:[], length:N}`.

---

## 1) ObservationPacket

Emitted by the YCB-Habitat wrapper each tick. Contains per-column crops and minimal per-crop egocentric pose.

### Example

```json
{
  "type": "observation.v1",
  "columns": [
    {
      "column_id": "col0",
      "view_id": "obj_008:orbitA",
      "patch": {
        "dtype": "uint8",
        "shape": [64, 64, 4],
        "storage": "shm://obs/col0/000123" 
      },
      "channels": ["rgb", "depth"],
      "egopose": { "u": 0.314, "v": 0.812, "u_prev": 0.280, "v_prev": 0.812 }
    }
  ],
  "global_meta": {
    "object_id": "obj_008_pudding_box",
    "tick": 123,
    "camera_intr": [575.8, 575.8, 320.0, 240.0]
  }
}
```

> `patch.storage` is an opaque handle (shared memory, mmap, file path). If you inline arrays for debugging, keep the same keys and add `"data": [ ... ]`.

### Schema (stub)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ObservationPacket",
  "type": "object",
  "required": ["type", "columns", "global_meta"],
  "properties": {
    "type": { "const": "observation.v1" },
    "columns": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["column_id", "patch", "channels", "egopose"],
        "properties": {
          "column_id": { "type": "string" },
          "view_id": { "type": "string" },
          "patch": {
            "type": "object",
            "required": ["dtype", "shape", "storage"],
            "properties": {
              "dtype": { "type": "string" },
              "shape": {
                "type": "array",
                "items": { "type": "integer" },
                "minItems": 3,
                "maxItems": 3
              },
              "storage": { "type": "string" }
            }
          },
          "channels": {
            "type": "array",
            "items": { "enum": ["rgb", "depth", "normals"] }
          },
          "egopose": {
            "type": "object",
            "required": ["u", "v", "u_prev", "v_prev"],
            "properties": {
              "u": { "type": "number" },
              "v": { "type": "number" },
              "u_prev": { "type": "number" },
              "v_prev": { "type": "number" }
            }
          }
        }
      }
    },
    "global_meta": {
      "type": "object",
      "required": ["object_id", "tick"],
      "properties": {
        "object_id": { "type": "string" },
        "tick": { "type": "integer" },
        "camera_intr": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 4,
          "maxItems": 4
        }
      }
    }
  },
  "additionalProperties": false
}
```

---

## 2) ContextPacket

Sparse binary control bus driving priors and gating learning.

### Example

```json
{
  "type": "context.v1",
  "c_bits": { "indices": [0, 1, 67], "length": 1024 },
  "sources": ["metronome", "switch", "intent"],
  "annotations": { "metronome_bit": 0, "switch_bit": 1 }
}
```

### Schema (stub)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ContextPacket",
  "type": "object",
  "required": ["type", "c_bits"],
  "properties": {
    "type": { "const": "context.v1" },
    "c_bits": {
      "type": "object",
      "required": ["indices", "length"],
      "properties": {
        "indices": {
          "type": "array",
          "items": { "type": "integer", "minimum": 0 },
          "uniqueItems": true
        },
        "length": { "type": "integer", "minimum": 1 }
      }
    },
    "sources": {
      "type": "array",
      "items": { "type": "string" }
    },
    "annotations": { "type": "object" }
  },
  "additionalProperties": false
}
```

---

    **Scenario-specific sources**

    * **Examiner** — morphological + non-morphological descriptors of the current object and the switch pulse bit.
    * **Explorer** — Examiner bits plus head-direction bins, motion-rate flags, intent toggles, and metronome clocks.
    * **Goalseeker** *(planned)* — Explorer bits plus reward-state encoding (recent reward sign/magnitude, goal-progress buckets).

    All variants share the same sparse structure; only the active indices differ.

    ---

## 3) PosePacket

For PI velocity derivation on a unit torus.

### Example

```json
{
  "type": "pose.v1",
  "per_column": [
    {
      "column_id": "col0",
      "pose_t":   { "u": 0.314, "v": 0.812 },
      "pose_tm1": { "u": 0.280, "v": 0.812 }
    }
  ],
  "dt": 0.05
}
```

### Schema (stub)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PosePacket",
  "type": "object",
  "required": ["type", "per_column", "dt"],
  "properties": {
    "type": { "const": "pose.v1" },
    "per_column": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["column_id", "pose_t", "pose_tm1"],
        "properties": {
          "column_id": { "type": "string" },
          "pose_t":   { "type": "object", "required": ["u","v"], "properties": { "u": { "type":"number" }, "v": { "type":"number" } } },
          "pose_tm1": { "type": "object", "required": ["u","v"], "properties": { "u": { "type":"number" }, "v": { "type":"number" } } }
        }
      }
    },
    "dt": { "type": "number", "exclusiveMinimum": 0 }
  },
  "additionalProperties": false
}
```

---

## 4) ActionMessage

Agent control with optional intent passthrough to context.

### Example

```json
{
  "type": "action.v1",
  "action_type": "move",
  "params": { "dx": 1.0, "dy": 0.0 },
  "intent_bits": { "indices": [3], "length": 1024 }
}
```

### Schema (stub)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ActionMessage",
  "type": "object",
  "required": ["type", "action_type", "params"],
  "properties": {
    "type": { "const": "action.v1" },
    "action_type": { "enum": ["move", "switch_object", "jump_to", "noop"] },
    "params": { "type": "object" },
    "intent_bits": {
      "type": "object",
      "required": ["indices", "length"],
      "properties": {
        "indices": { "type": "array", "items": { "type": "integer", "minimum": 0 }, "uniqueItems": true },
        "length": { "type": "integer", "minimum": 1 }
      }
    }
  },
  "additionalProperties": false
}
```

---

## 5) BeliefPacket

Motor-facing summary of shared and per-column beliefs plus helpers.

### Example

```json
{
  "type": "belief.v1",
  "g_star_logits": { "dtype": "float32", "shape": [2048], "storage": "shm://belief/g_star/000123" },
  "g_star_sdr": { "indices": [12, 97, 402, 955, 1311], "length": 2048 },
  "entropy": 0.78,
  "peakiness": 0.62,
  "per_column": {
    "col0": {
      "g_post_logits": { "dtype": "float32", "shape": [2048], "storage": "shm://belief/col0/g_post/000123" },
      "g_sdr": { "indices": [12, 97, 955], "length": 2048 },
      "f_sdr": { "indices": [44, 203, 511, 912], "length": 4096 }
    }
  },
  "c_sdr": { "indices": [0, 1, 67], "length": 1024 }
}
```

### Schema (stub)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "BeliefPacket",
  "type": "object",
  "required": ["type", "g_star_logits", "g_star_sdr", "per_column", "entropy", "peakiness", "c_sdr"],
  "properties": {
    "type": { "const": "belief.v1" },
    "g_star_logits": {
      "type": "object",
      "required": ["dtype","shape","storage"],
      "properties": {
        "dtype": { "type": "string" },
        "shape": { "type": "array", "items": { "type": "integer" } },
        "storage": { "type": "string" }
      }
    },
    "g_star_sdr": {
      "type": "object",
      "required": ["indices","length"],
      "properties": {
        "indices": { "type": "array", "items": { "type":"integer", "minimum": 0 }, "uniqueItems": true },
        "length": { "type": "integer", "minimum": 1 }
      }
    },
    "entropy": { "type": "number" },
    "peakiness": { "type": "number" },
    "per_column": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["g_post_logits","g_sdr","f_sdr"],
        "properties": {
          "g_post_logits": {
            "type": "object",
            "required": ["dtype","shape","storage"],
            "properties": {
              "dtype": { "type": "string" },
              "shape": { "type": "array", "items": { "type": "integer" } },
              "storage": { "type": "string" }
            }
          },
          "g_sdr": {
            "type": "object",
            "required": ["indices","length"],
            "properties": {
              "indices": { "type": "array", "items": { "type":"integer", "minimum": 0 }, "uniqueItems": true },
              "length": { "type": "integer", "minimum": 1 }
            }
          },
          "f_sdr": {
            "type": "object",
            "required": ["indices","length"],
            "properties": {
              "indices": { "type": "array", "items": { "type":"integer", "minimum": 0 }, "uniqueItems": true },
              "length": { "type": "integer", "minimum": 1 }
            }
          }
        }
      }
    },
    "c_sdr": {
      "type": "object",
      "required": ["indices","length"],
      "properties": {
        "indices": { "type": "array", "items": { "type":"integer", "minimum": 0 }, "uniqueItems": true },
        "length": { "type": "integer", "minimum": 1 }
      }
    }
  },
  "additionalProperties": false
}
```

---

## 6) FacetRecord

For counterfactual 2.5D facet synthesis and evaluation.

### Example

```json
{
  "type": "facet.v1",
  "phase_idx": 955,
  "coords_uv": [0.314, 0.812],
  "pred": { "dtype": "float32", "shape": [32, 32], "storage": "shm://facet/pred/000123" },
  "gt":   { "dtype": "float32", "shape": [32, 32], "storage": "shm://facet/gt/000123" },
  "losses": { "L1": 0.042, "PSNR": 27.9 }
}
```

### Schema (stub)

```json
{
  "$schema":"http://json-schema.org/draft-07/schema#",
  "title":"FacetRecord",
  "type":"object",
  "required":["type","phase_idx","pred","gt","losses"],
  "properties":{
    "type":{ "const":"facet.v1" },
    "phase_idx":{ "type":"integer", "minimum":0 },
    "coords_uv":{
      "type":"array", "items":{ "type":"number" }, "minItems":2, "maxItems":2
    },
    "pred":{
      "type":"object",
      "required":["dtype","shape","storage"],
      "properties":{
        "dtype":{"type":"string"},
        "shape":{"type":"array","items":{"type":"integer"}},
        "storage":{"type":"string"}
      }
    },
    "gt":{
      "type":"object",
      "required":["dtype","shape","storage"],
      "properties":{
        "dtype":{"type":"string"},
        "shape":{"type":"array","items":{"type":"integer"}},
        "storage":{"type":"string"}
      }
    },
    "losses":{
      "type":"object",
      "properties":{
        "L1":{"type":"number"},
        "PSNR":{"type":"number"}
      },
      "additionalProperties": {"type": ["number","integer","string","boolean"]}
    }
  },
  "additionalProperties": false
}
```

---

## 7) EvalRecord

Episode/experiment roll-ups for quick plotting and CI gating.

### Example

```json
{
  "type": "eval.v1",
  "episode_id": "2025-10-07T13:45:12Z#seed42",
  "metrics": {
    "entropy_mean": 0.81,
    "peakiness_mean": 0.54,
    "facet_L1_mean": 0.051,
    "coverage": 0.37
  },
  "series": {
    "entropy_ts": { "dtype": "float32", "shape": [500], "storage": "file://runs/exp1/entropy.bin" }
  }
}
```

### Schema (stub)

```json
{
  "$schema":"http://json-schema.org/draft-07/schema#",
  "title":"EvalRecord",
  "type":"object",
  "required":["type","episode_id","metrics"],
  "properties":{
    "type":{ "const":"eval.v1" },
    "episode_id":{ "type":"string" },
    "metrics":{ "type":"object" },
    "series":{
      "type":"object",
      "additionalProperties":{
        "type":"object",
        "required":["dtype","shape","storage"],
        "properties":{
          "dtype":{"type":"string"},
          "shape":{"type":"array","items":{"type":"integer"}},
          "storage":{"type":"string"}
        }
      }
    }
  },
  "additionalProperties": false
}
```

---

## Field Notes (sharp edges)

* **Torus deltas:** compute shortest signed delta on ([0,1)) per axis before scaling.
* **Top-K invariants:** `indices` sorted ascending, unique; budgets enforced in tests.
* **Storage indirection:** use handles for large arrays; keep inlined `data` only for debugging.
* **Versioning:** bump the `"type"` suffix (e.g., `observation.v2`) for breaking changes only.

---

## Validation & CI

* Each schema above should live under `contracts/*.schema.json`.
* `tests/test_packets_schema.py` validates every emitted packet with `jsonschema`.
* The smoke loop test must **fail** if unknown fields are present or required fields are missing.

---

## Quick Reference (copy/paste helpers)

Sparse vector helper (Python):

```python
def make_sparse(indices, length):
    return {"indices": sorted(set(map(int, indices))), "length": int(length)}
```

Array handle helper:

```python
def handle(dtype, shape, storage):
    return {"dtype": str(dtype), "shape": list(map(int, shape)), "storage": str(storage)}
```

---

If you want a `PACKETS.md` → `*.schema.json` generator script, I can add a tiny Python tool that writes these stubs and a validator CLI so CI can run `pytest -q && python tools/validate_packets.py`.
