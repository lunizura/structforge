from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from structforge import __version__
from structforge.binary_loader import load_binary_file
from structforge.codegen_cpp import generate_cpp_header
from structforge.codegen_py import generate_python_sdk
from structforge.codegen_tools import generate_cheat_engine_xml, generate_reclass_xml
from structforge.errors import InferenceError, StructForgeError
from structforge.infer import WatchSpec, synthesize_sdk
from structforge.models import FieldKind, SdkSpec
from structforge.oracle import verify_sdk
from structforge.report_html import generate_html_report
from structforge.rtti import harvest_rtti
from structforge.snapshot import load_snapshot


def _parse_watch_value(kind: FieldKind, raw: str) -> Any:
    if kind in (FieldKind.F32, FieldKind.F64):
        try:
            return float(raw)
        except ValueError as exc:
            raise InferenceError(f"Invalid float watch value '{raw}'") from exc
    if kind == FieldKind.BOOL:
        low = raw.strip().lower()
        if low in ("1", "true", "yes"):
            return True
        if low in ("0", "false", "no"):
            return False
        raise InferenceError(f"Invalid bool watch value '{raw}'")
    if kind == FieldKind.CSTR_PTR:
        return raw
    try:
        return int(raw, 0)
    except ValueError as exc:
        raise InferenceError(f"Invalid integer watch value '{raw}'") from exc


def _parse_watch_spec(spec_str: str, num_snaps: int) -> WatchSpec:
    parts = spec_str.split(":", 2)
    if len(parts) != 3:
        raise InferenceError(f"Invalid --watch '{spec_str}' (expected name:kind:v0,v1,...)")
    name, kind_raw, vals_raw = parts
    try:
        kind = FieldKind(kind_raw)
    except ValueError as exc:
        raise InferenceError(f"Unknown FieldKind '{kind_raw}' in --watch") from exc
    val_strs = [v.strip() for v in vals_raw.split(",")]
    if len(val_strs) != num_snaps:
        raise InferenceError(
            f"--watch '{name}' supplied {len(val_strs)} values for {num_snaps} snapshots"
        )
    values = tuple(_parse_watch_value(kind, v) for v in val_strs)
    return WatchSpec(name=name, kind=kind, values=values)


def _parse_struct_target(spec_str: str, num_snaps: int) -> tuple[str, list[int], int]:
    parts = spec_str.split(":", 2)
    if len(parts) != 3:
        raise InferenceError(f"Invalid --struct '{spec_str}' (expected Name:va0,va1,...:size)")
    name, vas_raw, size_raw = parts
    try:
        vas = [int(v.strip(), 0) for v in vas_raw.split(",") if v.strip()]
        size = int(size_raw, 0)
    except ValueError as exc:
        raise InferenceError(f"Invalid VA or size in --struct '{spec_str}'") from exc
    if len(vas) == 1 and num_snaps > 1:
        vas = vas * num_snaps
    if len(vas) != num_snaps:
        raise InferenceError(f"--struct '{name}' supplied {len(vas)} VAs for {num_snaps} snapshots")
    return name, vas, size


def _write_output(text: str, output_path: str | None) -> None:
    if output_path is None or output_path == "-":
        sys.stdout.write(text)
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def _cmd_rtti(args: argparse.Namespace) -> int:
    binary = load_binary_file(args.binary)
    vtables = harvest_rtti(binary)
    if args.json:
        payload = json.dumps([vt.to_dict() for vt in vtables], indent=2) + "\n"
        _write_output(payload, args.output)
        return 0
    lines = [
        f"Format: {binary.format} | ImageBase: 0x{binary.image_base:X} | VTables: {len(vtables)}"
    ]
    for vt in vtables:
        bases = ", ".join(vt.base_classes) if vt.base_classes else "-"
        lines.append(
            f"  0x{vt.vtable_rva:06X}  {vt.demangled_name:<28}  methods={len(vt.method_rvas)}  bases={bases}"
        )
    _write_output("\n".join(lines) + "\n", args.output)
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    snapshots = [load_snapshot(p) for p in args.snapshots]
    binary = load_binary_file(args.binary) if args.binary else None
    watches = [_parse_watch_spec(w, len(snapshots)) for w in (args.watch or [])]
    targets = [_parse_struct_target(t, len(snapshots)) for t in (args.struct or [])]
    root_va = int(args.root_va, 0) if args.root_va else None
    spec, oracle = synthesize_sdk(
        snapshots=snapshots,
        binary=binary,
        watches=watches,
        name=args.name,
        module_name=args.module,
        root_va=root_va,
        struct_targets=targets,
        max_depth=args.max_depth,
        max_span=args.max_span,
    )
    _write_output(spec.to_json(), args.output)
    if args.report:
        _write_output(generate_html_report(spec, snapshots=snapshots, oracle=oracle), args.report)
    if args.output and args.output != "-":
        print(
            f"Synthesized '{spec.name}': {len(spec.structs)} structs, {len(spec.pointer_chains)} chains, "
            f"score={oracle.layout_score:.2f} (passed={oracle.passed})",
            file=sys.stderr,
        )
    if not oracle.passed:
        for fail in oracle.failures:
            print(f"oracle failure: {fail}", file=sys.stderr)
        return 1
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    spec = SdkSpec.from_json(Path(args.spec).read_text(encoding="utf-8"))
    snapshots = [load_snapshot(p) for p in args.snapshots]
    binary = load_binary_file(args.binary) if args.binary else None
    res = verify_sdk(spec, snapshots, binary=binary, trials=args.trials, seed=args.seed)
    if args.json:
        payload = {
            "passed": res.passed,
            "gate1_abi_layout": res.gate1_abi_layout,
            "gate2_snapshot_replay": res.gate2_snapshot_replay,
            "gate3_aslr_and_drift": res.gate3_aslr_and_drift,
            "aslr_pass_rate": res.aslr_pass_rate,
            "drift_reanchor_rate": res.drift_reanchor_rate,
            "layout_score": res.layout_score,
            "failures": res.failures,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        print(f"Gate 1 ABI Layout:     {'PASS' if res.gate1_abi_layout else 'FAIL'}")
        print(f"Gate 2 Snapshot Replay:{'PASS' if res.gate2_snapshot_replay else 'FAIL'}")
        print(f"Gate 3 ASLR & Drift:   {'PASS' if res.gate3_aslr_and_drift else 'FAIL'}")
        print(
            f"ASLR={res.aslr_pass_rate:.2%} | Drift={res.drift_reanchor_rate:.2%} | "
            f"Layout Score={res.layout_score:.2f}/100"
        )
        for fail in res.failures:
            print(f"  - {fail}", file=sys.stderr)
    return 0 if res.passed else 1


def _cmd_compile(args: argparse.Namespace) -> int:
    spec = SdkSpec.from_json(Path(args.spec).read_text(encoding="utf-8"))
    if args.lang == "cpp":
        out = generate_cpp_header(spec)
    elif args.lang == "python":
        out = generate_python_sdk(spec)
    elif args.lang == "reclass":
        out = generate_reclass_xml(spec)
    elif args.lang == "ce":
        out = generate_cheat_engine_xml(spec)
    else:
        raise InferenceError(f"Unsupported compile language '{args.lang}'")
    _write_output(out, args.output)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    spec = SdkSpec.from_json(Path(args.spec).read_text(encoding="utf-8"))
    snapshots = [load_snapshot(p) for p in (args.snapshots or [])]
    binary = load_binary_file(args.binary) if args.binary else None
    oracle = verify_sdk(spec, snapshots, binary=binary) if snapshots else None
    html_out = generate_html_report(spec, snapshots=snapshots, oracle=oracle)
    _write_output(html_out, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structforge",
        description="PE32+/ELF64 RTTI harvester, multi-snapshot struct recoverer & C++20 SDK compiler",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rtti = sub.add_parser("rtti", help="Harvest MSVC x64 or Itanium ABI RTTI vtables")
    p_rtti.add_argument("binary", help="Path to PE32+ (.dll/.exe) or ELF64 binary")
    p_rtti.add_argument("--json", action="store_true", help="Emit JSON output")
    p_rtti.add_argument("-o", "--output", help="Output file path (default: stdout)")
    p_rtti.set_defaults(func=_cmd_rtti)

    p_synth = sub.add_parser("synthesize", help="Recover structs and pointer chains from snapshots")
    p_synth.add_argument(
        "--snapshots", nargs="+", required=True, help="Ordered .sfdump snapshot files"
    )
    p_synth.add_argument("--binary", help="Optional PE32+/ELF64 binary for RTTI enrichment")
    p_synth.add_argument(
        "--watch",
        action="append",
        help="Differential watch spec: name:kind:v0,v1,... (e.g. health:f32:100.0,72.5,0.0)",
    )
    p_synth.add_argument(
        "--struct",
        action="append",
        help="Explicit struct target: Name:va0,va1,...:size",
    )
    p_synth.add_argument("--name", default="GeneratedSdk", help="SDK identifier")
    p_synth.add_argument("--module", default="module.dll", help="Module name for CE/comments")
    p_synth.add_argument("--root-va", help="Override snapshot root VA (hex or dec)")
    p_synth.add_argument("--max-depth", type=int, default=4, help="Max BFS pointer depth")
    p_synth.add_argument(
        "--max-span", type=lambda s: int(s, 0), default=0x200, help="Max scan span"
    )
    p_synth.add_argument("-o", "--output", help="Output SdkSpec JSON path")
    p_synth.add_argument("--report", help="Optional HTML5 report output path")
    p_synth.set_defaults(func=_cmd_synthesize)

    p_verify = sub.add_parser("verify", help="Run 3-Gate ABI, Replay & ASLR/Drift Oracle")
    p_verify.add_argument("spec", help="Path to SdkSpec JSON")
    p_verify.add_argument(
        "--snapshots", nargs="+", required=True, help="Ordered .sfdump snapshot files"
    )
    p_verify.add_argument("--binary", help="Optional PE32+/ELF64 binary")
    p_verify.add_argument("--trials", type=int, default=100, help="Number of ASLR fuzz trials")
    p_verify.add_argument(
        "--seed", type=lambda s: int(s, 0), default=0x5746049E, help="Fuzzer seed"
    )
    p_verify.add_argument("--json", action="store_true", help="Emit JSON result")
    p_verify.set_defaults(func=_cmd_verify)

    p_compile = sub.add_parser("compile", help="Compile SdkSpec JSON to C++20/Python/ReClass/CE")
    p_compile.add_argument("spec", help="Path to SdkSpec JSON")
    p_compile.add_argument(
        "--lang",
        required=True,
        choices=("cpp", "python", "reclass", "ce"),
        help="Target output language/format",
    )
    p_compile.add_argument("-o", "--output", help="Output file path (default: stdout)")
    p_compile.set_defaults(func=_cmd_compile)

    p_inspect = sub.add_parser("inspect", help="Generate dark-mode HTML5 Struct & VTable inspector")
    p_inspect.add_argument("spec", help="Path to SdkSpec JSON")
    p_inspect.add_argument(
        "--snapshots", nargs="*", help="Optional .sfdump snapshots for live replay"
    )
    p_inspect.add_argument("--binary", help="Optional PE32+/ELF64 binary")
    p_inspect.add_argument("-o", "--output", help="Output HTML file path (default: stdout)")
    p_inspect.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (StructForgeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
