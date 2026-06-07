"""Serving retriever: goal signature constant NAMES -> ranked premises.
Graph-free: the context vector is one context_gnn pass over a tiny subgraph
(involved signature premises -> 1 context node), ensembled over the 8 models.
Loads only the precomputed artifacts (no dataset, no full models, no leandojo)."""
import os
import torch
import gnn_model

_HERE = os.path.dirname(os.path.abspath(__file__))


class EnsembleGNNRetriever:
    def __init__(self, artifacts=None, device="cpu"):
        artifacts = artifacts or os.path.join(_HERE, "artifacts")
        self.dev = torch.device(device)
        # embedding tables are stored fp16 (small download); use fp32 for compute
        self.premise_emb = torch.load(f"{artifacts}/premise_emb.pt").float().to(self.dev)   # (M,N,H)
        self.rinit = torch.load(f"{artifacts}/rand_init_ctx.pt").float().to(self.dev)        # (M,N,H)
        states = torch.load(f"{artifacts}/ctx_gnn_states.pt")
        cfgc = torch.load(f"{artifacts}/ctx_gnn_config.pt")
        self.gnns = []
        for st in states:
            gnn = gnn_model.GNN(cfgc); gnn.load_state_dict(st); gnn.eval(); gnn.to(self.dev)
            self.gnns.append(gnn)
        meta = torch.load(f"{artifacts}/meta.pt")
        self.idx2name = meta["idx2name"]; self.name2idx = meta["name2idx"]
        self.GOAL = meta["goal_id"]; self.LCTX = meta["lctx_id"]
        self.H = meta["hidden"]; self.M = len(self.gnns)

    @torch.no_grad()
    def score_vector(self, goal_names, lctx_names):
        gi = [self.name2idx[n] for n in goal_names if n in self.name2idx]
        li = [self.name2idx[n] for n in lctx_names if n in self.name2idx]
        dropped = [n for n in list(goal_names) + list(lctx_names) if n not in self.name2idx]
        inv = sorted(set(gi) | set(li))
        pos = {p: k for k, p in enumerate(inv)}
        cl = len(inv)
        src, dst, et = [], [], []
        for p in gi: src.append(pos[p]); dst.append(cl); et.append(self.GOAL)
        for p in li: src.append(pos[p]); dst.append(cl); et.append(self.LCTX)
        ei = torch.tensor([src, dst], dtype=torch.long, device=self.dev) if src \
            else torch.zeros((2, 0), dtype=torch.long, device=self.dev)
        ett = torch.tensor(et, dtype=torch.long, device=self.dev) if et \
            else torch.zeros((0,), dtype=torch.long, device=self.dev)
        total = None
        for i in range(self.M):
            base = self.rinit[i][inv] if inv else torch.empty(0, self.H, device=self.dev)
            feats = torch.cat([base, torch.zeros(1, self.H, device=self.dev)], dim=0)
            cv = self.gnns[i](feats, ei, ett)[cl]
            s = self.premise_emb[i] @ cv
            total = s if total is None else total + s
        return total / self.M, dropped

    @torch.no_grad()
    def retrieve(self, goal_names, lctx_names, k=16):
        scores, dropped = self.score_vector(goal_names, lctx_names)
        vals, idx = torch.topk(scores, min(k, scores.shape[0]))
        return ([{"name": self.idx2name[int(j)], "score": float(v)} for v, j in zip(vals, idx)],
                dropped)
