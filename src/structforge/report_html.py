from __future__ import annotations

import html
from collections.abc import Sequence

from structforge.codegen_cpp import generate_cpp_header
from structforge.models import FieldKind, SdkSpec, StructDef, validate_spec
from structforge.oracle import OracleResult, verify_sdk
from structforge.snapshot import HeapSnapshot


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _kind_color(kind: FieldKind) -> str:
    if kind == FieldKind.VPTR:
        return "#a855f7"
    if kind in (FieldKind.PTR64, FieldKind.CSTR_PTR, FieldKind.STD_VECTOR):
        return "#38bdf8"
    if kind in (FieldKind.F32, FieldKind.F64):
        return "#22c55e"
    if kind in (
        FieldKind.I32,
        FieldKind.U32,
        FieldKind.I64,
        FieldKind.U64,
        FieldKind.U16,
        FieldKind.U8,
    ):
        return "#f59e0b"
    if kind == FieldKind.BOOL:
        return "#ec4899"
    if kind == FieldKind.CHAR_ARRAY:
        return "#14b8a6"
    return "#475569"


def _cpp_display_type(kind: FieldKind, size: int) -> str:
    if kind == FieldKind.CHAR_ARRAY:
        return f"char[{size}]"
    if kind == FieldKind.PAD:
        return f"uint8_t[{size}]"
    return {
        FieldKind.VPTR: "uint64_t (vptr)",
        FieldKind.PTR64: "uint64_t*",
        FieldKind.CSTR_PTR: "const char*",
        FieldKind.STD_VECTOR: "RawVectorView",
        FieldKind.F64: "double",
        FieldKind.I64: "int64_t",
        FieldKind.U64: "uint64_t",
        FieldKind.F32: "float",
        FieldKind.I32: "int32_t",
        FieldKind.U32: "uint32_t",
        FieldKind.U16: "uint16_t",
        FieldKind.U8: "uint8_t",
        FieldKind.BOOL: "bool",
    }[kind]


def _render_chain_svg(spec: SdkSpec) -> str:
    if not spec.pointer_chains:
        return '<svg width="640" height="80" xmlns="http://www.w3.org/2000/svg"><text x="16" y="42" fill="#94a3b8" font-size="13">No pointer chains defined.</text></svg>'

    row_h = 74
    max_hops = max(len(c.offsets) for c in spec.pointer_chains)
    width = max(720, 220 + (max_hops + 1) * 165)
    height = 24 + len(spec.pointer_chains) * row_h

    parts: list[str] = [
        f'<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'style="background:#0b1220;border-radius:8px;border:1px solid #334155;">'
    ]
    for r_idx, ch in enumerate(spec.pointer_chains):
        y = 18 + r_idx * row_h
        x = 16
        # Root box
        parts.append(
            f'<rect x="{x}" y="{y}" width="136" height="44" rx="6" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{x + 68}" y="{y + 18}" fill="#e2e8f0" font-size="11" text-anchor="middle" font-family="monospace">{_e(spec.module_name)}</text>'
        )
        parts.append(
            f'<text x="{x + 68}" y="{y + 34}" fill="#38bdf8" font-size="11" text-anchor="middle" font-family="monospace">+0x{ch.root_rva:X}</text>'
        )
        x += 136

        for h_idx, off in enumerate(ch.offsets):
            is_leaf = h_idx == len(ch.offsets) - 1
            parts.append(
                f'<line x1="{x}" y1="{y + 22}" x2="{x + 36}" y2="{y + 22}" stroke="#64748b" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{x + 18}" y="{y + 15}" fill="#fbbf24" font-size="10" text-anchor="middle" font-family="monospace">+0x{off:X}</text>'
            )
            x += 36
            stroke = "#22c55e" if is_leaf else "#6366f1"
            label_top = (
                ch.name
                if is_leaf
                else (
                    ch.target_struct
                    if h_idx == len(ch.offsets) - 2 and ch.target_struct
                    else f"Hop {h_idx + 1}"
                )
            )
            label_bot = ch.field_kind.value if is_leaf else "ptr64"
            parts.append(
                f'<rect x="{x}" y="{y}" width="128" height="44" rx="6" fill="#1e293b" stroke="{stroke}" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{x + 64}" y="{y + 18}" fill="#f8fafc" font-size="11" text-anchor="middle" font-family="monospace">{_e(label_top)}</text>'
            )
            parts.append(
                f'<text x="{x + 64}" y="{y + 34}" fill="{stroke}" font-size="10" text-anchor="middle" font-family="monospace">{_e(label_bot)}</text>'
            )
            x += 128

    parts.append("</svg>")
    return "".join(parts)


def _render_struct_card(st: StructDef) -> str:
    rows: list[str] = []
    for fld in st.fields:
        cache_line = fld.offset // 64
        row_8b = fld.offset // 8
        color = _kind_color(fld.kind)
        if fld.sample_values:
            transitions = " &rarr; ".join(_e(v) for v in fld.sample_values[:5])
        else:
            transitions = "&mdash;"
        rtti_note = ""
        if fld.target_type:
            rtti_note = f' <span class="badge-muted">{_e(fld.target_type)}</span>'
        rows.append(
            "<tr>"
            f'<td class="mono">CL{cache_line} / R{row_8b}</td>'
            f'<td class="mono">+0x{fld.offset:04X}</td>'
            f'<td><span class="pill" style="border-color:{color};color:{color};">{_e(_cpp_display_type(fld.kind, fld.size))}</span></td>'
            f'<td class="mono">{_e(fld.name)}{rtti_note}</td>'
            f'<td class="mono">{fld.size} B</td>'
            f'<td class="mono">{transitions}</td>'
            "</tr>"
        )

    rtti_line = (
        f'<span class="badge-purple">RTTI: {_e(st.demangled_rtti)}</span>'
        if st.demangled_rtti
        else '<span class="badge-muted">No RTTI</span>'
    )
    bases_line = (
        f'<span class="badge-muted">Bases: {_e(", ".join(st.base_classes))}</span>'
        if st.base_classes
        else ""
    )
    return (
        '<div class="card">'
        f'<div class="card-head"><h3>{_e(st.name)}</h3>'
        f'<div><span class="badge-info">sizeof = 0x{st.size:X} ({st.size} B)</span> '
        f'<span class="badge-info">align = {st.alignment}</span> {rtti_line} {bases_line}</div></div>'
        '<table class="data-table"><thead><tr>'
        "<th>Cache / 8B Row</th><th>Offset</th><th>C++ Type</th><th>Field</th><th>Size</th><th>Snapshot Transitions</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def generate_html_report(
    spec: SdkSpec,
    snapshots: Sequence[HeapSnapshot] = (),
    oracle: OracleResult | None = None,
) -> str:
    validate_spec(spec)
    if oracle is None and snapshots:
        oracle = verify_sdk(spec, snapshots)

    score = oracle.layout_score if oracle is not None else spec.layout_score

    def _badge(label: str, ok: bool | None) -> str:
        if ok is None:
            return f'<span class="badge-muted">{_e(label)}: N/A</span>'
        cls = "badge-pass" if ok else "badge-fail"
        status = "PASS" if ok else "FAIL"
        return f'<span class="{cls}">{_e(label)}: {status}</span>'

    badges_html = "".join(
        [
            _badge("Gate 1 ABI Layout", oracle.gate1_abi_layout if oracle else None),
            _badge("Gate 2 Snapshot Replay", oracle.gate2_snapshot_replay if oracle else None),
            _badge("Gate 3 ASLR & Drift", oracle.gate3_aslr_and_drift if oracle else None),
            f'<span class="badge-score">Layout Score: {score:.2f} / 100</span>',
        ]
    )

    vtable_rows: list[str] = []
    for vt in spec.vtables:
        methods = ", ".join(f"0x{rva:X}" for rva in vt.method_rvas[:6])
        if len(vt.method_rvas) > 6:
            methods += f" (+{len(vt.method_rvas) - 6})"
        vtable_rows.append(
            "<tr>"
            f'<td class="mono">{_e(vt.demangled_name)}</td>'
            f'<td class="mono">{_e(vt.mangled_name)}</td>'
            f'<td class="mono">{_e(", ".join(vt.base_classes) or "-")}</td>'
            f'<td class="mono">0x{vt.vtable_rva:X}</td>'
            f'<td class="mono">{_e(methods)}</td>'
            "</tr>"
        )
    vtable_html = (
        "".join(vtable_rows)
        if vtable_rows
        else '<tr><td colspan="5" class="muted">No RTTI vtables recorded.</td></tr>'
    )

    cpp_preview = _e(generate_cpp_header(spec))
    structs_html = "\n".join(_render_struct_card(st) for st in spec.structs)
    svg_html = _render_chain_svg(spec)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>structforge Inspector &mdash; {_e(spec.name)}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: 1.5rem; color: #f8fafc; }}
    h2 {{ margin: 0 0 12px 0; font-size: 1.15rem; color: #f1f5f9; }}
    h3 {{ margin: 0; font-size: 1.05rem; color: #f8fafc; }}
    .muted {{ color: var(--muted); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85rem; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .badge-pass, .badge-fail, .badge-score, .badge-info, .badge-purple, .badge-muted {{
      display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
    }}
    .badge-pass {{ background: #14532d; color: #86efac; border: 1px solid #22c55e; }}
    .badge-fail {{ background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }}
    .badge-score {{ background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }}
    .badge-info {{ background: #0f172a; color: #38bdf8; border: 1px solid #0284c7; }}
    .badge-purple {{ background: #3b0764; color: #d8b4fe; border: 1px solid #a855f7; }}
    .badge-muted {{ background: #0f172a; color: var(--muted); border: 1px solid var(--border); }}
    .card {{
      background: var(--card); border: 1px solid var(--border); border-radius: 10px;
      padding: 16px; margin-bottom: 18px; overflow-x: auto;
    }}
    .card-head {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 12px; }}
    .data-table {{ width: 100%; border-collapse: collapse; }}
    .data-table th, .data-table td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
    .data-table th {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    .pill {{ display: inline-block; padding: 1px 8px; border: 1px solid; border-radius: 6px; font-family: monospace; font-size: 0.8rem; }}
    pre {{ margin: 0; padding: 14px; background: #0b1220; border: 1px solid var(--border); border-radius: 8px; overflow-x: auto; font-size: 0.82rem; color: #cbd5e1; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>structforge Inspector: {_e(spec.name)}</h1>
        <div class="muted mono">{_e(spec.binary_format)} &bull; {_e(spec.module_name)} &bull; {_e(spec.description)}</div>
      </div>
      <div class="badges">{badges_html}</div>
    </div>

    <div class="card">
      <h2>Pointer-Chain Hop Diagram</h2>
      {svg_html}
    </div>

    <div class="card">
      <h2>RTTI Inheritance &amp; VTable Map</h2>
      <table class="data-table">
        <thead>
          <tr><th>Demangled Class</th><th>Mangled Symbol</th><th>Base Classes</th><th>VTable RVA</th><th>Method RVAs</th></tr>
        </thead>
        <tbody>{vtable_html}</tbody>
      </table>
    </div>

    {structs_html}

    <div class="card">
      <h2>Generated C++20 SDK Header Preview</h2>
      <pre><code>{cpp_preview}</code></pre>
    </div>
  </div>
</body>
</html>
"""
