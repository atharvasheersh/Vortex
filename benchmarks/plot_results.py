"""Plot saved measurements without manufacturing benchmark values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory, evaluation=None):
    directory = Path(directory)
    rows = json.loads((directory/"measurements.json").read_text())
    faiss = json.loads((directory/"faiss.json").read_text())
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
    selected = [r for r in rows if r["factor"] == "nprobe"]
    ax = axes[0,0]
    ax.plot([r["p95_ms"] for r in selected],[r["ann_recall"] for r in selected],"o-",label="Vortex IVF")
    selected_faiss = [r for r in faiss if r["family"] == "ivf"]
    if selected_faiss: ax.plot([r["p95_ms"] for r in selected_faiss],[r["ann_recall"] for r in selected_faiss],"s--",label="FAISS IVF")
    ax.set(xlabel="p95 retrieval wall time (ms)",ylabel="ANN Recall@10",title="Quality and latency")
    ax.legend()
    selected = [r for r in rows if r["factor"] == "threads"]
    axes[0,1].plot([r["threads"] for r in selected],[r["batch_qps"] for r in selected],"o-")
    axes[0,1].set(xlabel="Workers",ylabel="Batch queries / second",title="Exact batch throughput")
    selected = [r for r in rows if r["factor"] == "representation"]
    axes[0,2].bar([r["setting"] for r in selected],[r["bytes_per_vector"] for r in selected])
    axes[0,2].set(ylabel="Total snapshot bytes / vector",title="Original FP32 vectors retained")
    selected = [r for r in rows if r["factor"] == "cache_budget"]
    axes[1,0].plot([r["cache_budget"] for r in selected],[r["p95_ms"] for r in selected],"o-")
    axes[1,0].set(xlabel="Application cache payload budget (bytes)",ylabel="p95 (ms)",title="OS page cache uncontrolled",xscale="symlog")
    selected = [r for r in rows if r["factor"] == "corpus_size"]
    axes[1,1].plot([r["vectors"] for r in selected],[r["p95_ms"] for r in selected],"o-")
    axes[1,1].set(xlabel="Vectors",ylabel="p95 exact search (ms)",title="Collection scaling")
    selected = [r for r in rows if r["factor"] == "cache_state"]
    axes[1,2].bar([r["setting"] for r in selected],[r["p95_ms"] for r in selected])
    axes[1,2].set(ylabel="p95 (ms)",title="Application cache state")
    fig.suptitle("Vortex controlled synthetic measurements",fontsize=15)
    fig.savefig(directory/"systems.png",dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(14,4),layout="constrained")
    selected = [r for r in rows if r["factor"] == "cache_budget"]
    rates = [r["hits"]/(r["hits"]+r["misses"]) if r["hits"]+r["misses"] else 0 for r in selected]
    axes[0].plot([r["cache_budget"] for r in selected],rates,"o-")
    axes[0].set(xlabel="Cache budget (bytes)",ylabel="Block hit rate",xscale="symlog",title="Application cache only")
    selected = [r for r in rows if r["factor"] == "corpus_size"]
    axes[1].plot([r["vectors"] for r in selected],[r["build_seconds"] for r in selected],"o-")
    axes[1].set(xlabel="Vectors",ylabel="IVF build seconds",title="Fixed lists and iterations")
    selected = [r for r in rows if r["factor"] == "representation"]
    axes[2].bar([r["setting"] for r in selected],[r["ann_recall"] for r in selected])
    axes[2].set(ylabel="ANN Recall@10",ylim=(0,1.05),title="All lists; no reranking")
    fig.savefig(directory/"storage-build.png",dpi=180)
    plt.close(fig)
    if evaluation:
        data = json.loads(Path(evaluation).read_text())
        fig,axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
        labels = [str(r["nprobe"]) for r in data]
        axes[0].plot(labels,[r["retrieval_metrics"]["recall"] for r in data],"o-",label="Relevant-chunk recall")
        axes[0].plot(labels,[r["answer_metrics"]["exact_match"] for r in data],"s--",label="Extractive fixture exact match")
        axes[0].set(xlabel="nprobe (0 = exact)",ylabel="Score",ylim=(0,1.05))
        axes[0].legend(fontsize=8)
        axes[1].plot([r["p95_retrieval_ms"] for r in data],
                     [r["answer_metrics"]["exact_match"] for r in data],"o-")
        axes[1].set(xlabel="p95 retrieval wall time (ms)",ylabel="Extractive fixture exact match")
        fig.suptitle("Small illustrative fixture; no LLM quality claim")
        fig.savefig(directory/"fixture-quality.png",dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",default="results/benchmark")
    p.add_argument("--evaluation",default=None)
    args = p.parse_args()
    plot(args.directory,args.evaluation)
