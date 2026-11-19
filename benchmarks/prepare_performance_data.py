"""Generate optimization inputs with the existing bounded version-1 writer."""

import argparse
import json
import shutil
from pathlib import Path

from capacity_resources import working_set_limit
from run_capacity import check_resources
from streaming_snapshot import build_snapshot, layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=32)
    parser.add_argument("--training-limit", type=int, default=10000)
    parser.add_argument("--working-set-mib", type=int, default=1536)
    args = parser.parse_args()
    if (
        args.count < args.nlist
        or args.nlist < 1
        or args.dimension < 1
        or args.training_limit < args.nlist
    ):
        parser.error(
            "require positive dimensions/lists, count >= nlist, training_limit >= nlist"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    required = layout(args.count, args.dimension, args.nlist)[-1] + args.count * 4
    required += layout(
        min(args.count, args.training_limit), args.dimension, args.nlist
    )[-1]
    if shutil.disk_usage(args.output.parent).free < required + 256 * 1024**2:
        raise OSError(
            "insufficient disk space for snapshot, assignments, training data and headroom"
        )
    with working_set_limit(args.working_set_mib):
        result = build_snapshot(
            args.output,
            args.count,
            args.dimension,
            args.nlist,
            training_limit=args.training_limit,
            check_resources=check_resources,
        )
    (args.output / "build.json").write_bytes(
        (json.dumps(result, indent=2) + "\n").encode()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
