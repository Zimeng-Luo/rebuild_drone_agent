"""Preview or remove generated reconstruction results; inputs and weights stay intact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOTS = (ROOT / "data" / "result", ROOT / "experiment" / "result")


def result_inventory() -> list[dict[str, object]]:
    inventory = []
    for directory in RESULT_ROOTS:
        if directory.is_symlink():
            raise RuntimeError(f"Refusing symlink result directory: {directory}")
        resolved = directory.resolve()
        if resolved != directory or not resolved.is_relative_to(ROOT):
            raise RuntimeError(f"Unexpected result directory: {resolved}")
        if not directory.exists():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.name == ".gitkeep":
                continue
            size = 0
            if not entry.is_symlink():
                if entry.is_file():
                    size = entry.stat().st_size
                else:
                    size = sum(p.stat().st_size for p in entry.rglob("*")
                               if p.is_file() and not p.is_symlink())
            inventory.append({"path": str(entry), "bytes": size})
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete listed generated results")
    parser.add_argument("--manifest", type=Path, help="Write the inventory before deletion")
    args = parser.parse_args()
    inventory = result_inventory()
    total = sum(int(item["bytes"]) for item in inventory)
    print(json.dumps({"apply": args.apply, "total_bytes": total, "entries": inventory}, indent=2))
    if args.manifest:
        if args.manifest.resolve().is_relative_to(ROOT / "data" / "result") or args.manifest.resolve().is_relative_to(ROOT / "experiment" / "result"):
            parser.error("The manifest must be outside directories being cleaned")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"total_bytes": total, "entries": inventory}, indent=2), encoding="utf-8")
    if args.apply:
        for item in inventory:
            path = Path(str(item["path"]))
            if path.parent not in RESULT_ROOTS:
                raise RuntimeError(f"Unexpected deletion target: {path}")
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
        print(f"Removed {len(inventory)} generated entries ({total / 1024 ** 3:.2f} GiB).")
    else:
        print("Preview only. Add --apply to remove these generated results.")


if __name__ == "__main__":
    main()
