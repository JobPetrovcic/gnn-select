# gnn_select

A Lean tactic that suggests premises using a GNN ensemble. A Python service holds
the model; the Lean tactic extracts the goal's constants and queries it over HTTP.

## Requirements
- Python 3.10–3.12, `git`, `curl`
- Lean toolchain `leanprover/lean4:v4.10.0-rc1`
- mathlib commit `29dcec074de168ac2bf835a77ef68bbe069194c5`

## Setup
```bash
git clone https://github.com/JobPetrovcic/gnn-select
cd gnn-select
bash setup.sh        # Python venv + deps + downloads weights + self-test
bash setup_lean.sh   # Lean toolchain + pinned mathlib + builds GnnSelect.lean
```

## Run
Start the service (leave running):
```bash
source .venv/bin/activate
python service/serve.py            # http://127.0.0.1:8077
```
Use the tactic in the bundled project — open `lean/gnn_lean_project` in VSCode
(Lean4 extension), open `GnnSelect.lean`, place the cursor after `gnn_select`;
or `cd lean/gnn_lean_project && lake env lean GnnSelect.lean`.

## Use in your own project
Your project must use the pinned versions above (toolchain `v4.10.0-rc1`,
mathlib `29dcec07…`). Then:
1. Copy the tactic definitions from `lean/GnnSelect.lean` (everything above the
   `example`) into a `.lean` file in your project.
2. Start the service (above).
3. Use `gnn_select` in any proof.
