import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from vortex import Index
from vortex.rag import SentenceEmbedder, http_answer


def test_http_generation_contract():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"response":"Seven days."}).encode())
        def log_message(self, *_args):
            pass
    server = HTTPServer(("127.0.0.1",0),Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        result = http_answer("How long?","Exports last seven days.",
                             f"http://127.0.0.1:{server.server_port}/generate","fixture-server")
        assert result == "Seven days."
        assert "Exports last seven days." in requests[0]["prompt"]
        assert requests[0]["options"] == {"temperature":0,"seed":42}
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_pretrained_embedding_requires_pinned_model():
    with pytest.raises(ValueError):
        SentenceEmbedder("some-model",None)


def test_uint64_ids_and_empty_snapshot(tmp_path):
    index = Index(2)
    path = tmp_path/"empty.vtx"
    index.save(str(path))
    opened = Index.open(str(path))
    assert len(opened) == 0
    assert opened.search(np.ones(2,dtype=np.float32))["hits"] == []
    index.add((1<<64)-1,np.ones(2,dtype=np.float32))
    index.add(0,np.ones(2,dtype=np.float32))
    assert [h[0] for h in index.search(np.ones(2,dtype=np.float32))["hits"]] == [0,(1<<64)-1]


def test_fp16_failure_preserves_previous_representation():
    index = Index(2)
    vector = np.array([70000,0],dtype=np.float32)
    index.add(1,vector)
    with pytest.raises(ValueError):
        index.set_dtype("fp16")
    assert index.dtype == "fp32"
    assert index.search(vector)["hits"] == [(1,0)]


@pytest.mark.parametrize("dtype",["fp16","int8"])
def test_lazy_nonfinite_payload_rejection(dtype,tmp_path):
    import struct
    index = Index(2)
    index.add(1,np.ones(2,dtype=np.float32))
    index.set_dtype(dtype)
    path = tmp_path/f"{dtype}.vtx"
    index.save(str(path))
    raw = bytearray(path.read_bytes())
    offset = struct.unpack_from("<Q",raw,56)[0]
    if dtype == "fp16": struct.pack_into("<H",raw,offset,0x7c00)
    else: struct.pack_into("<f",raw,offset,float("nan"))
    path.write_bytes(raw)
    opened = Index.open(str(path))
    with pytest.raises(RuntimeError):
        opened.search(np.ones(2,dtype=np.float32))


def test_probe_coverage_and_recall_are_monotonic():
    rng = np.random.default_rng(52)
    index = Index(9)
    for identifier,vector in enumerate(rng.normal(size=(300,9)).astype(np.float32)):
        index.add(identifier,vector)
    index.build_ivf(12)
    for query in rng.normal(size=(10,9)).astype(np.float32):
        truth = {h[0] for h in index.search(query)["hits"]}
        scanned,hit_count = 0,0
        for nprobe in range(1,13):
            result = index.search(query,nprobe=nprobe)
            current = len(truth & {h[0] for h in result["hits"]})
            assert result["scanned"] >= scanned
            assert current >= hit_count
            scanned,hit_count = result["scanned"],current
        assert scanned == 300 and hit_count == 10
