"""Zero-dependency HTTP server (stdlib only) for the GNN premise retriever.
Run:  python service/serve.py
Env:  GNN_SVC_PORT (default 8077), GNN_SVC_DEVICE (cpu|cuda:0)
  POST /retrieve {"goal_constants":[...], "lctx_constants":[...], "k":16}
  GET  /health
"""
import os, sys, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retriever import EnsembleGNNRetriever

R = EnsembleGNNRetriever(device=os.environ.get("GNN_SVC_DEVICE", "cpu"))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "models": R.M, "premises": len(R.idx2name)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/retrieve":
            self._send(404, {"error": "not found"}); return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        results, dropped = R.retrieve(body.get("goal_constants", []),
                                      body.get("lctx_constants", []),
                                      int(body.get("k", 16)))
        self._send(200, {"results": results, "dropped_unknown": dropped})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("GNN_SVC_PORT", "8077"))
    print(f"[serve] GNN retriever on http://127.0.0.1:{port}  "
          f"(device={os.environ.get('GNN_SVC_DEVICE','cpu')}, models={R.M}, premises={len(R.idx2name)})",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
