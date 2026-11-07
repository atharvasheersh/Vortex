"""Render the editable Vortex architecture plate and its README preview."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ASSETS = Path(__file__).resolve().parents[1] / "assets"
INK = "#122C42"
MUTED = "#385363"
TEAL = "#006E67"
LINE = "#718B95"
PAPER = "#FBFCFA"
WHITE = "#FFFFFF"


def render():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "vortex-architecture",
        }
    )
    fig = plt.figure(figsize=(12, 7.1), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 1200), ylim=(710, 0))
    ax.set_axis_off()

    def label(x, y, value, size=11, *, color=INK, bold=False):
        ax.text(
            x,
            y,
            value,
            va="center",
            fontsize=size,
            color=color,
            weight="bold" if bold else "normal",
            linespacing=1.3,
            zorder=4,
        )

    def panel(x, y, width, height, fill=WHITE, edge="#C5D3D5"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=9",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.9,
            )
        )

    def arrow(points):
        # A single head marks the destination, including on elbow connectors.
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=TEAL, linewidth=1.5, solid_capstyle="round")
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.5,
                color=TEAL,
                shrinkA=0,
                shrinkB=1,
                zorder=3,
            )
        )

    def card(x, y, width, title, detail, *, fill=WHITE):
        panel(x, y, width, 76, fill)
        label(x + 15, y + 25, title, 14, bold=True)
        label(x + 15, y + 54, detail, 10.7, color=MUTED)

    # Align the input and output lanes with the dispatch and ranking stages.
    ax.add_patch(Rectangle((30, 23), 6, 42, color=TEAL))
    label(48, 44, "VORTEX", 25, bold=True)
    label(239, 45, "Vector retrieval architecture", 22)
    ax.plot([30, 1170], [83, 83], color=LINE, linewidth=0.7)

    label(30, 123, "INPUTS", 11, color=TEAL, bold=True)
    card(30, 156, 237, "Chunks + question", "Corpus text or query text")
    arrow([(148, 232), (148, 270)])
    panel(30, 270, 237, 105)
    label(45, 296, "Embedding model", 14, bold=True)
    label(45, 341, "Pretrained model or\noffline hash encoder", color=MUTED)
    arrow([(267, 321), (290, 321), (290, 194), (332, 194)])

    panel(311, 99, 556, 381, fill="#E7F3EF", edge="#84AFA5")
    label(332, 123, "C++20 RETRIEVAL ENGINE", 11, color=TEAL, bold=True)
    card(
        332,
        156,
        514,
        "Python and C++ API",
        "Validate vectors  /  dispatch batch queries to workers",
    )
    arrow([(452, 232), (452, 263)])
    arrow([(722, 232), (722, 263)])
    card(332, 263, 248, "Exact search", "Score every vector")
    card(598, 263, 248, "IVF search", "Probe selected centroid lists")
    arrow([(452, 339), (452, 373)])
    arrow([(722, 339), (722, 373)])
    card(
        332,
        373,
        514,
        "Ranked results",
        "Stable top-k IDs  /  optional rerank with FP32 originals",
    )

    label(929, 123, "ANSWER PIPELINE", 11, color=TEAL, bold=True)
    arrow([(846, 411), (893, 411), (893, 194), (929, 194)])
    card(929, 156, 241, "Chunk lookup", "Resolve ranked IDs to text")
    arrow([(1049, 232), (1049, 263)])
    card(929, 263, 241, "Generation", "API or extractive fixture")
    arrow([(1049, 339), (1049, 373)])
    card(929, 373, 241, "Answer + evidence", "Log results for each query")

    # This band describes the backing store, outside the query sequence.
    arrow([(589, 504), (589, 480)])
    panel(30, 504, 1140, 73, fill=INK, edge=INK)
    label(49, 527, "PERSISTENT STORAGE", 10.5, color="#A4E3CD", bold=True)
    label(49, 554, "Mapped .vtx snapshot", 15, color=WHITE, bold=True)
    label(378, 527, "Vectors and index", 13, color=WHITE, bold=True)
    label(378, 553, "IDs, FP32 originals, centroids and postings", 10.6, color=WHITE)
    label(819, 527, "Encoded reads", 13, color=WHITE, bold=True)
    label(819, 553, "FP32 / FP16 / INT8  ·  bounded LRU cache", 10.6, color=WHITE)

    label(30, 609, "EVALUATION", 11, color=TEAL, bold=True)
    ax.plot([30, 1170], [627, 627], color=LINE, linewidth=0.7)
    for x, title, detail in [
        (30, "Retrieval quality", "Recall@k and nDCG against labels"),
        (422, "Answer diagnostics", "Token F1 and evidence text overlap"),
        (811, "System benchmarks", "Latency, QPS, RSS and FAISS baselines"),
    ]:
        label(x, 650, title, 14, bold=True)
        label(x, 679, detail, 11, color=MUTED)

    ASSETS.mkdir(exist_ok=True)
    fig.savefig(
        ASSETS / "system-architecture.png", dpi=200, metadata={"Software": "Vortex"}
    )
    svg = ASSETS / "system-architecture.svg"
    fig.savefig(svg, metadata={"Creator": "Vortex", "Date": None})
    # Normalize the SVG backend's platform newlines and path padding for Git.
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text(
        "".join(line.rstrip() + "\n" for line in lines), encoding="utf-8", newline="\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    render()
