"""Generate the editable system diagram and its README PNG."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def render():
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none"})
    fig,ax = plt.subplots(figsize=(16,10))
    fig.patch.set_facecolor("#f6f8fc")
    ax.set(xlim=(0,16),ylim=(0,10))
    ax.axis("off")
    ax.text(.45,9.5,"VORTEX",fontsize=26,weight="bold",color="#14233d")
    ax.text(.45,9.08,"Persistent vector retrieval with reproducible quality and systems evaluation",fontsize=13,color="#455a75")
    boundary = FancyBboxPatch((4.55,2.25),7.25,6.2,boxstyle="round,pad=0.12",linewidth=1.5,
                             edgecolor="#91a8c5",facecolor="#eaf0fa")
    ax.add_patch(boundary)
    ax.text(4.8,8.1,"C++20 ENGINE",fontsize=12,weight="bold",color="#294f83")
    def box(x,y,w,h,title,detail,color="#ffffff"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",linewidth=1,
                                   edgecolor="#bdc9da",facecolor=color,zorder=3))
        ax.text(x+w/2,y+h*.66,title,ha="center",va="center",fontsize=14,weight="bold",color="#14233d",zorder=4)
        ax.text(x+w/2,y+h*.27,detail,ha="center",va="center",fontsize=12,color="#455a75",zorder=4)
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,linewidth=1.3,color="#566d8c",zorder=2))
        if label:
            ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,label,ha="center",fontsize=9,color="#455a75")
    box(.5,7.15,3.15,1,"Documents and chunks","Offline text preparation")
    box(.5,5.45,3.15,1,"Embedding model","External pretrained model\nor explicit offline fixture")
    box(.5,3.75,3.15,1,"Question","Natural-language query")
    box(5,6.65,6.3,1,"Python and C++ API","Validated inputs  |  Query engine  |  Worker pool")
    box(5,4.95,2.8,1,"Exact search","Exhaustive top-k")
    box(8.5,4.95,2.8,1,"IVF search","Centroids + nprobe lists")
    box(5,3.5,6.3,.9,"Candidates and optional reranking","Bounded heap  |  Stable IDs  |  FP32 originals")
    box(5,2.5,6.3,.65,"Storage access","FP32 / FP16 / INT8  |  Bounded LRU block cache")
    box(12.5,6.65,3,1,"Chunk lookup","IDs resolve to text")
    box(12.5,4.95,3,1,"Generation","External model endpoint\nor extractive fixture")
    box(12.5,3.5,3,.9,"Answer and evidence","Per-query result log")
    box(5,.85,6.3,1,"Versioned mapped snapshot","Stable IDs + original vectors + encoded vectors\nCentroids + list offsets + postings")
    box(.5,.85,3.15,1,"Evaluation harness","Labels + ranking metrics\nAnswer diagnostics + manifest")
    box(12.5,.85,3,1,"Systems measurements","Latency / QPS / cache / RSS\nFAISS baselines and plots")
    arrow((2.075,7.15),(2.075,6.45))
    arrow((3.65,5.95),(5,6.95),"embeddings")
    arrow((2.075,4.75),(2.075,5.45))
    arrow((6.4,6.65),(6.4,5.95)); arrow((9.9,6.65),(9.9,5.95))
    arrow((6.4,4.95),(6.4,4.4)); arrow((9.9,4.95),(9.9,4.4))
    arrow((8.15,3.5),(8.15,3.15)); arrow((8.15,2.5),(8.15,1.85))
    arrow((11.3,6.95),(12.5,6.95),"top-k IDs")
    arrow((14,6.65),(14,5.95)); arrow((14,4.95),(14,4.4))
    arrow((11.8,2.75),(12.1,2.75))
    arrow((12.1,2.75),(12.1,1.35))
    arrow((12.1,1.35),(12.5,1.35))
    arrow((14,3.5),(14,2.1))
    arrow((14,2.1),(2.075,2.1),"evidence and answers to evaluation")
    arrow((2.075,2.1),(2.075,1.85))
    ax.text(.5,.35,"External models are separate from the engine. Offline fixtures verify plumbing and do not establish LLM quality.",
            fontsize=10,color="#455a75")
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
    target = Path("assets")
    target.mkdir(exist_ok=True)
    fig.savefig(target/"system-architecture.png",dpi=170,facecolor=fig.get_facecolor(),metadata={"Software":"Vortex"})
    fig.savefig(target/"system-architecture.svg",facecolor=fig.get_facecolor(),metadata={"Creator":"Vortex","Date":None})
    plt.close(fig)


if __name__ == "__main__": render()
