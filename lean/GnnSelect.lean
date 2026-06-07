import Mathlib
open Lean Elab Tactic Meta

set_option linter.unusedTactic false  -- gnn_select only logs; it doesn't change the goal

/-- Verbose signature extractor (LeanDojo-faithful): walk the Expr, collect every `.const`. -/
partial def extractAllConsts (e : Expr) : MetaM (Array Name) := do
  let r ← IO.mkRef (#[] : Array Name)
  let visitor (s : Expr) : MetaM Bool := do
    if let .const n _ := s then r.modify (·.push n)
    return true
  e.forEach' visitor
  r.get

/-- POST the goal's signature constants to the GNN retrieval service. -/
def queryGNNService (goal lctx : Array Name) (k : Nat) : IO String := do
  let port := (← IO.getEnv "GNN_SVC_PORT").getD "8077"
  let req := Json.mkObj [
    ("goal_constants", Json.arr (goal.map (fun n => Json.str (toString n)))),
    ("lctx_constants", Json.arr (lctx.map (fun n => Json.str (toString n)))),
    ("k", toJson k)]
  IO.Process.run {
    cmd := "curl",
    args := #["-s", "--fail", s!"http://127.0.0.1:{port}/retrieve",
              "-H", "Content-Type: application/json", "-d", req.compress] }

/-- `gnn_select`: extract the current goal's signature premises (goal + local context)
    and log the GNN model's top suggestions. Requires the Python service running. -/
elab "gnn_select" : tactic => do
  let g ← getMainGoal
  let decl ← g.getDecl
  Meta.withLCtx decl.lctx decl.localInstances do
    let goalC ← extractAllConsts (← instantiateMVars decl.type)
    let lctxC ← decl.lctx.foldlM (init := (#[] : Array Name)) fun acc d => do
      if !d.isAuxDecl ∧ !d.isImplementationDetail then return acc ++ (← extractAllConsts d.type)
      else return acc
    let resp ← (do queryGNNService goalC lctxC 10) <|>
               (do logWarning "[gnn_select] could not reach service — is service/serve.py running?"; pure "")
    if resp.isEmpty then return
    match Json.parse resp >>= (·.getObjVal? "results") >>= (·.getArr?) with
    | .ok results =>
      let mut lines : Array String := #[]
      for r in results do
        let nm := ((r.getObjVal? "name") >>= (·.getStr?)).toOption.getD "?"
        lines := lines.push s!"  • {nm}"
      logInfo s!"gnn_select — top {lines.size} premises:\n{String.intercalate "\n" lines.toList}"
    | .error e => logWarning s!"[gnn_select] bad service response: {e}\n{resp}"

/- =========================================================================
   DEMO — with the service running, place the cursor after `gnn_select` to see
   suggestions in the InfoView. Edit the statement to try your own goals.
   ========================================================================= -/
example {α β M : Type*} [Zero M] (f : α ↪ β) :
    Function.Injective (Finsupp.embDomain (M := M) f) := by
  gnn_select
  intro l₁ l₂ h
  ext a
  simpa only [Finsupp.embDomain_apply] using DFunLike.ext_iff.1 h (f a)
