from __future__ import annotations

from pathlib import Path

from structforge.snapshot import save_snapshot
from structforge.synthetic_fixtures import make_demo_suite


def main() -> None:
    samples_dir = Path(__file__).resolve().parent / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    suite = make_demo_suite()
    (samples_dir / "game_rtti.dll").write_bytes(suite.pe_bytes)
    (samples_dir / "engine_rtti.elf").write_bytes(suite.elf_bytes)
    for idx, snap in enumerate(suite.snapshots):
        save_snapshot(snap, samples_dir / f"snap_{idx}.sfdump")

    print(f"Generated demo binaries and 3 snapshots in {samples_dir}")


if __name__ == "__main__":
    main()
