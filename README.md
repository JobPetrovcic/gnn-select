# gnn_select — pure-GNN premise selection for Lean

A Lean tactic, `gnn_select`, that suggests premises using a pure-GNN ensemble
(8 models) trained on the LeanDojo Benchmark 4 `random` split. Two parts:

- **Python service** (`service/`): holds the model, scores premises. Pure `torch` +
  `torch_geometric` — **no leandojo, no dataset, no GPU required.**
- **Lean tactic** (`lean/GnnSelect.lean`): extracts the goal's constants and asks the service.

## Requirements

- **Python 3.10–3.12** (developed on 3.11).
- **No GPU needed** — the bundled torch is the **CPU build**; runs entirely on CPU (~70 ms/query).
  (To run on a GPU you'd reinstall the CUDA torch build and set `GNN_SVC_DEVICE=cuda:0` — unnecessary.)
- **RAM:** ~6 GB (measured 5.9 GB — the fp16 embedding tables are cast to fp32 in memory).
- **Disk:** ~2.8 GB (the `service/artifacts/` embeddings, shipped fp16).
- `curl` on PATH (used by the Lean tactic to call the service).
- A **Lean 4 + Mathlib** project. Best results on the corpus snapshot
  **mathlib commit `29dcec07…`, toolchain `leanprover/lean4:v4.10.0-rc1`**
  (on a different mathlib version, constants that aren't in the corpus are dropped —
  see `dropped_unknown` in the response — and quality degrades).

## Install & self-test (one command)

```bash
bash setup.sh
```
This creates `.venv`, installs CPU `torch`+`torch_geometric`, starts the service,
and verifies a known query returns `Finsupp.embDomain_apply`.

## Use it

1. **Start the service** (leave running):
   ```bash
   source .venv/bin/activate
   python service/serve.py            # http://127.0.0.1:8077  (set GNN_SVC_PORT to change)
   ```
2. **Add the tactic to your Lean project:** copy `lean/GnnSelect.lean` into a
   Lean+Mathlib project (ideally the snapshot above), open it, and use `gnn_select`
   in a proof. With the cursor after `gnn_select`, the InfoView shows the top premises.

   Quick batch check (no editor): from your mathlib project dir,
   ```bash
   lake env lean /path/to/GnnSelect.lean
   ```

## Test the service directly (no Lean)

```bash
curl -s localhost:8077/retrieve -H 'Content-Type: application/json' \
  -d '{"goal_constants":["Function.Injective","Finsupp.embDomain"],"lctx_constants":[],"k":10}'
```

## Notes
- The model is the GNN ensemble only (no LM); it ranks premises that are
  structurally related (dependency-graph neighbors) to the goal's constants.
- It can only retrieve premises in the corpus snapshot (~180k mathlib/core decls);
  premises added later are out of scope (a fixed-snapshot limitation).
- Smaller footprint: a single-model variant (~370 MB) is possible if 2.8 GB is too
  large to ship — ask the author to re-export with one model.
