from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from structforge.cli import main as cli_main
from structforge.codegen_cpp import generate_cpp_header
from structforge.codegen_py import generate_python_sdk
from structforge.codegen_tools import generate_cheat_engine_xml, generate_reclass_xml
from structforge.infer import WatchSpec, synthesize_sdk
from structforge.models import FieldKind, SdkSpec, StructDef, StructField
from structforge.report_html import generate_html_report
from structforge.synthetic_fixtures import make_demo_suite


def _find_cpp20_compiler(tmp_path: Path) -> tuple[str, str]:
    """Return (kind, path) where kind is 'gcc_like' or 'msvc_vcvars'."""
    probe_cpp = tmp_path / "_probe_cpp20.cpp"
    probe_exe = tmp_path / ("_probe_cpp20.exe" if os.name == "nt" else "_probe_cpp20")
    probe_cpp.write_text(
        "#include <concepts>\n#include <span>\n"
        "template <typename T> concept C = std::integral<T>;\n"
        "int main() { return 0; }\n",
        encoding="utf-8",
    )

    for compiler in ("g++", "clang++"):
        path = shutil.which(compiler)
        if path is None:
            continue
        proc = subprocess.run(
            [
                path,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(probe_cpp),
                "-o",
                str(probe_exe),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return ("gcc_like", path)

    if os.name == "nt":
        roots = [
            Path("C:/Program Files/Microsoft Visual Studio"),
            Path("C:/Program Files (x86)/Microsoft Visual Studio"),
        ]
        for root in roots:
            for year in ("2022", "2019"):
                for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
                    cand = root / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    if cand.is_file():
                        return ("msvc_vcvars", str(cand))

    pytest.fail("No C++20 compiler (g++/clang++ or VS vcvars64.bat) available")
    raise AssertionError("unreachable")


def _format_cpp_byte_array(var_name: str, data: bytes) -> str:
    hex_bytes = ", ".join(f"0x{b:02x}u" for b in data)
    return f"static constexpr std::array<std::uint8_t, {len(data)}> {var_name}{{{{{hex_bytes}}}}};"


def test_live_cpp20_compile_execute_and_multiheader_odr(tmp_path: Path) -> None:
    suite = make_demo_suite()
    watches = [
        WatchSpec(name="health", kind=FieldKind.F32, values=(100.0, 72.5, 0.0)),
        WatchSpec(name="ammo", kind=FieldKind.I32, values=(30, 29, 0)),
        WatchSpec(name="is_alive", kind=FieldKind.BOOL, values=(True, True, False)),
    ]
    spec, oracle = synthesize_sdk(
        snapshots=suite.snapshots,
        binary=suite.pe_binary,
        watches=watches,
        name="GameSdk",
        module_name="game_rtti.dll",
    )
    assert oracle.passed is True

    aux_spec = SdkSpec(
        name="AuxSdk",
        binary_format="raw64",
        module_name="aux.dll",
        structs=[
            StructDef(
                name="AuxRecord",
                size=16,
                alignment=8,
                fields=[
                    StructField(name="id", offset=0, size=4, kind=FieldKind.I32),
                    StructField(name="ratio", offset=4, size=4, kind=FieldKind.F32),
                    StructField(name="tag_ptr", offset=8, size=8, kind=FieldKind.PTR64),
                ],
            )
        ],
    )

    game_hpp = tmp_path / "game_sdk.hpp"
    aux_hpp = tmp_path / "aux_sdk.hpp"
    game_hpp.write_text(generate_cpp_header(spec), encoding="utf-8")
    aux_hpp.write_text(generate_cpp_header(aux_spec), encoding="utf-8")

    # Verify both headers have the ODR guard
    assert "#ifndef STRUCTFORGE_GEN_RUNTIME_HELPERS" in game_hpp.read_text(encoding="utf-8")
    assert "#ifndef STRUCTFORGE_GEN_RUNTIME_HELPERS" in aux_hpp.read_text(encoding="utf-8")

    snap0 = suite.snapshots[0]
    snap1 = suite.snapshots[1]

    cpp_runner = tmp_path / "test_runner.cpp"
    cpp_src = f"""
#include "game_sdk.hpp"
#include "aux_sdk.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>

using namespace structforge_gen;

{_format_cpp_byte_array("s0_data", snap0.regions[0].data)}
{_format_cpp_byte_array("s0_heap", snap0.regions[1].data)}
{_format_cpp_byte_array("s1_data", snap1.regions[0].data)}
{_format_cpp_byte_array("s1_heap", snap1.regions[1].data)}

int main() {{
    const std::array<RegionView, 2> r0{{{{
        RegionView{{0x{snap0.regions[0].base_va:x}ULL, std::span<const std::uint8_t>{{s0_data}}}},
        RegionView{{0x{snap0.regions[1].base_va:x}ULL, std::span<const std::uint8_t>{{s0_heap}}}},
    }}}};
    const std::array<RegionView, 2> r1{{{{
        RegionView{{0x{snap1.regions[0].base_va:x}ULL, std::span<const std::uint8_t>{{s1_data}}}},
        RegionView{{0x{snap1.regions[1].base_va:x}ULL, std::span<const std::uint8_t>{{s1_heap}}}},
    }}}};

    const SpanSnapshotReader reader0{{r0}};
    const SpanSnapshotReader reader1{{r1}};
    constexpr std::uint64_t image_base = 0x{snap0.image_base:x}ULL;

    const auto hp0 = read_chain_health(reader0, image_base);
    const auto hp1 = read_chain_health(reader1, image_base);
    assert(hp0.has_value() && std::fabs(*hp0 - 100.0f) < 1e-4f);
    assert(hp1.has_value() && std::fabs(*hp1 - 72.5f) < 1e-4f);

    const auto ammo0 = read_chain_ammo(reader0, image_base);
    const auto ammo1 = read_chain_ammo(reader1, image_base);
    assert(ammo0.has_value() && *ammo0 == 30);
    assert(ammo1.has_value() && *ammo1 == 29);

    const auto alive0 = read_chain_is_alive(reader0, image_base);
    const auto alive1 = read_chain_is_alive(reader1, image_base);
    assert(alive0.has_value() && *alive0 == true);
    assert(alive1.has_value() && *alive1 == true);

    const auto player0 = read_val<PlayerEntity>(reader0, 0x{suite.player_vas[0]:x}ULL);
    assert(player0.has_value());
    assert(std::fabs(player0->health - 100.0f) < 1e-4f);
    assert(player0->ammo == 30);
    assert(player0->is_alive == true);
    assert(player0->vec_0028.count(4u) == 3u);

    const AuxRecord aux{{42, 1.5f, 0x1234ULL}};
    assert(aux.id == 42);
    return 0;
}}
"""
    cpp_runner.write_text(cpp_src, encoding="utf-8")

    # Also verify a third header exercising all 15 FieldKinds + comment newline sanitization
    all_kinds_spec = SdkSpec(
        name="AllKindsSdk",
        binary_format="raw64",
        module_name="all.dll\n#error INJECTED_COMMENT",
        structs=[
            StructDef(
                name="AllKinds",
                size=112,
                alignment=8,
                demangled_rtti="ns::AllKinds\n#error INJECTED_RTTI",
                base_classes=["Base\n#error INJECTED_BASE"],
                fields=[
                    StructField(
                        name="vptr",
                        offset=0x00,
                        size=8,
                        kind=FieldKind.VPTR,
                        vtable_rva=0x1000,
                        target_type="ns::AllKinds\n#error TT",
                    ),
                    StructField(
                        name="ptr",
                        offset=0x08,
                        size=8,
                        kind=FieldKind.PTR64,
                        target_type="Other\n#error PT",
                    ),
                    StructField(name="cstr", offset=0x10, size=8, kind=FieldKind.CSTR_PTR),
                    StructField(
                        name="vec", offset=0x18, size=24, kind=FieldKind.STD_VECTOR, elem_stride=4
                    ),
                    StructField(name="d", offset=0x30, size=8, kind=FieldKind.F64),
                    StructField(name="i64", offset=0x38, size=8, kind=FieldKind.I64),
                    StructField(name="u64", offset=0x40, size=8, kind=FieldKind.U64),
                    StructField(name="f32", offset=0x48, size=4, kind=FieldKind.F32),
                    StructField(name="i32", offset=0x4C, size=4, kind=FieldKind.I32),
                    StructField(name="u32", offset=0x50, size=4, kind=FieldKind.U32),
                    StructField(name="u16", offset=0x54, size=2, kind=FieldKind.U16),
                    StructField(name="u8", offset=0x56, size=1, kind=FieldKind.U8),
                    StructField(name="flag", offset=0x57, size=1, kind=FieldKind.BOOL),
                    StructField(name="label", offset=0x58, size=16, kind=FieldKind.CHAR_ARRAY),
                    StructField(name="_pad_tail", offset=0x68, size=8, kind=FieldKind.PAD),
                ],
            )
        ],
    )
    all_kinds_cpp = generate_cpp_header(all_kinds_spec)
    assert all(
        "//" in line and line.index("//") < line.index("INJECTED")
        for line in all_kinds_cpp.splitlines()
        if "INJECTED" in line
    )
    all_kinds_hpp = tmp_path / "all_kinds.hpp"
    all_kinds_hpp.write_text(all_kinds_cpp, encoding="utf-8")
    cpp_src = cpp_src.replace(
        '#include "aux_sdk.hpp"', '#include "aux_sdk.hpp"\n#include "all_kinds.hpp"'
    )
    cpp_src = cpp_src.replace(
        "const AuxRecord aux{{42, 1.5f, 0x1234ULL}};",
        "const AuxRecord aux{{42, 1.5f, 0x1234ULL}};\n    const AllKinds ak{};\n    assert(sizeof(ak) == 112u);",
    )
    cpp_runner.write_text(cpp_src, encoding="utf-8")

    kind, compiler_path = _find_cpp20_compiler(tmp_path)
    exe_path = tmp_path / ("test_runner.exe" if os.name == "nt" else "test_runner")

    if kind == "gcc_like":
        comp = subprocess.run(
            [
                compiler_path,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(cpp_runner),
                "-o",
                str(exe_path),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
    else:
        obj_path = tmp_path / "test_runner.obj"
        bat_path = tmp_path / "build_msvc.bat"
        bat_path.write_text(
            "@echo off\r\n"
            f'call "{compiler_path}" >nul\r\n'
            "if errorlevel 1 exit /b %errorlevel%\r\n"
            f'cl /nologo /std:c++latest /O2 /EHsc /W4 /WX "{cpp_runner}" /Fo:"{obj_path}" /Fe:"{exe_path}"\r\n',
            encoding="utf-8",
        )
        comp = subprocess.run(
            ["cmd.exe", "/c", str(bat_path)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    assert comp.returncode == 0, f"C++20 compile failed:\n{comp.stdout}\n{comp.stderr}"
    run = subprocess.run([str(exe_path)], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, f"C++20 runner failed:\n{run.stdout}\n{run.stderr}"


def test_python_reclass_ce_and_html_xss() -> None:
    suite = make_demo_suite()
    watches = [
        WatchSpec(name="health", kind=FieldKind.F32, values=(100.0, 72.5, 0.0)),
        WatchSpec(name="ammo", kind=FieldKind.I32, values=(30, 29, 0)),
        WatchSpec(name="is_alive", kind=FieldKind.BOOL, values=(True, True, False)),
        WatchSpec(
            name="hero_tag",
            kind=FieldKind.CSTR_PTR,
            values=("ACTUAL_HERO_01", "ACTUAL_HERO_01", "ACTUAL_HERO_01"),
        ),
        WatchSpec(name="inv_vec", kind=FieldKind.STD_VECTOR, values=(12, 16, 20)),
    ]
    spec, oracle = synthesize_sdk(
        snapshots=suite.snapshots,
        binary=suite.pe_binary,
        watches=watches,
        name="GameSdk",
        module_name="game_rtti.dll",
    )
    assert oracle.passed is True

    # 1. Execute generated Python SDK (including CSTR_PTR -> str and STD_VECTOR -> RawVectorView)
    py_code = generate_python_sdk(spec)
    assert "import structforge" not in py_code
    ns: dict[str, object] = {}
    exec(compile(py_code, "<generated_sdk>", "exec"), ns, ns)

    snap0 = suite.snapshots[0]
    hp = ns["resolve_chain_health"](snap0.read_va, snap0.image_base)  # type: ignore[operator]
    ammo = ns["resolve_chain_ammo"](snap0.read_va, snap0.image_base)  # type: ignore[operator]
    alive = ns["resolve_chain_is_alive"](snap0.read_va, snap0.image_base)  # type: ignore[operator]
    tag = ns["resolve_chain_hero_tag"](snap0.read_va, snap0.image_base)  # type: ignore[operator]
    inv = ns["resolve_chain_inv_vec"](snap0.read_va, snap0.image_base)  # type: ignore[operator]
    assert abs(float(hp) - 100.0) < 1e-4  # type: ignore[arg-type]
    assert ammo == 30
    assert alive is True
    assert tag == "ACTUAL_HERO_01"
    assert inv.byte_len == 12 and inv.count(4) == 3  # type: ignore[attr-defined]

    player_cls = ns["PlayerEntity"]
    player_bytes = snap0.read_va(suite.player_vas[0], player_cls.SIZE)  # type: ignore[attr-defined]
    assert player_bytes is not None
    player_obj = player_cls.from_bytes(player_bytes)  # type: ignore[attr-defined]
    assert abs(player_obj.health - 100.0) < 1e-4
    assert player_obj.ammo == 30
    assert player_obj.inv_vec.count(4) == 3
    with pytest.raises(ValueError):
        player_cls.from_bytes(player_bytes, offset=-4)  # type: ignore[attr-defined]

    # 2. Cheat Engine XML: bottom-up reverse offsets (10, 40, 18)
    ce_xml = generate_cheat_engine_xml(spec)
    ce_root = ET.fromstring(ce_xml)
    entries = ce_root.findall("./CheatEntries/CheatEntry")
    hp_entry = next(e for e in entries if e.findtext("Description") == '"health"')
    offsets = [el.text for el in hp_entry.findall("./Offsets/Offset")]
    assert offsets == ["10", "40", "18"]
    assert hp_entry.findtext("Address") == "game_rtti.dll+5000"

    # 3. ReClass.NET XML validity
    rc_xml = generate_reclass_xml(spec)
    rc_root = ET.fromstring(rc_xml)
    class_names = {c.attrib["Name"] for c in rc_root.findall("./Classes/Class")}
    assert {"WorldContext", "Session", "PlayerEntity"}.issubset(class_names)

    # 4. HTML5 Inspector XSS escaping
    spec.description = '<script>alert(1)</script> & "quoted"'
    spec.module_name = 'mod"><script>alert(2)</script>.dll'
    html_out = generate_html_report(spec, snapshots=suite.snapshots, oracle=oracle)
    assert "<script>alert(1)</script>" not in html_out
    assert "<script>alert(2)</script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
    assert "<svg " in html_out


def test_cli_end_to_end_all_subcommands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    samples = Path(__file__).resolve().parent.parent / "examples" / "samples"
    dll_path = samples / "game_rtti.dll"
    elf_path = samples / "engine_rtti.elf"
    s0 = samples / "snap_0.sfdump"
    s1 = samples / "snap_1.sfdump"
    s2 = samples / "snap_2.sfdump"
    assert dll_path.is_file() and elf_path.is_file() and s0.is_file()

    # 1. rtti
    assert cli_main(["rtti", str(dll_path), "--json"]) == 0
    rtti_json = json.loads(capsys.readouterr().out)
    assert len(rtti_json) == 3

    assert cli_main(["rtti", str(elf_path)]) == 0
    assert "game::PlayerEntity" in capsys.readouterr().out

    # 2. synthesize
    spec_path = tmp_path / "game_sdk.json"
    rc = cli_main(
        [
            "synthesize",
            "--snapshots",
            str(s0),
            str(s1),
            str(s2),
            "--binary",
            str(dll_path),
            "--module",
            "game_rtti.dll",
            "--name",
            "GameSdk",
            "--watch",
            "health:f32:100.0,72.5,0.0",
            "--watch",
            "ammo:i32:30,29,0",
            "--watch",
            "is_alive:bool:true,true,false",
            "-o",
            str(spec_path),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    assert spec_path.is_file()

    # 3. verify
    assert (
        cli_main(
            [
                "verify",
                str(spec_path),
                "--snapshots",
                str(s0),
                str(s1),
                str(s2),
                "--binary",
                str(dll_path),
                "--json",
            ]
        )
        == 0
    )
    verify_out = json.loads(capsys.readouterr().out)
    assert verify_out["passed"] is True
    assert verify_out["aslr_pass_rate"] == 1.0

    # 4. compile (all 4 languages)
    for lang, ext in (("cpp", "hpp"), ("python", "py"), ("reclass", "rcnet"), ("ce", "CT")):
        out_file = tmp_path / f"sdk.{ext}"
        assert cli_main(["compile", str(spec_path), "--lang", lang, "-o", str(out_file)]) == 0
        assert out_file.read_text(encoding="utf-8").strip()

    # 5. inspect
    report_path = tmp_path / "report.html"
    assert (
        cli_main(
            [
                "inspect",
                str(spec_path),
                "--snapshots",
                str(s0),
                str(s1),
                str(s2),
                "-o",
                str(report_path),
            ]
        )
        == 0
    )
    assert "<!DOCTYPE html>" in report_path.read_text(encoding="utf-8")

    # 6. CLI error handling -> exit code 1 (including missing files)
    assert cli_main(["rtti", str(tmp_path / "does_not_exist.dll")]) == 1
    assert "error:" in capsys.readouterr().err

    assert cli_main(["rtti", str(spec_path)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err

    assert (
        cli_main(
            [
                "synthesize",
                "--snapshots",
                str(s0),
                str(s1),
                "--watch",
                "ammo:i32:bad,val",
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err

    assert (
        cli_main(
            [
                "synthesize",
                "--snapshots",
                str(s0),
                str(s1),
                "--struct",
                "Player:not_hex,not_hex:16",
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err

    # Also check CLI entrypoint via subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "structforge.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "0.1.0" in proc.stdout
