# structforge

[![CI](https://github.com/lunizura/structforge/actions/workflows/ci.yml/badge.svg)](https://github.com/lunizura/structforge/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)
![License MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Targets](https://img.shields.io/badge/Targets-C%2B%2B20%20%7C%20Python%20%7C%20ReClass%20%7C%20CE-A855F7?style=flat-square)
![Dependencies](https://img.shields.io/badge/Runtime%20Deps-100%25%20stdlib-14B8A6?style=flat-square)

**`structforge`** is a zero-dependency Python (`>=3.10`) binary reverse-engineering toolkit and layout-verified C++20 SDK compiler. It harvests MSVC x64 (`COL_SIG_REV1`) and ELF64 Itanium (`_ZTV`/`_ZTI`) RTTI/VTable hierarchies, recovers struct fields and pointer chains across multiple `.sfdump` heap snapshots, proves ABI & ASLR resilience via a 3-Gate Oracle (`0–100`), and compiles `#pragma once` C++20 headers with compile-time `sizeof`/`offsetof` guarantees.

---

## Architecture

```text
  PE32+ (.dll/.exe) / ELF64                Multi-Snapshot Heap Dumps (.sfdump)
             │                                             │
             ▼                                             ▼
    [ binary_loader.py ]                         [ snapshot.py (SFMD v1) ]
             │                                             │
             ▼                                             │
    [ rtti.py: MSVC COL_SIG_REV1 / Itanium _ZTV+16 ]       │
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    ▼
             [ infer.py: Type Cascade + Bounded BFS Pointer Chains ]
                                    │
                                    ▼
             [ oracle.py: 3-Gate ABI Layout, Replay & 100-Trial ASLR/Drift ]
                                    │
         ┌──────────────────┬───────┴──────────┬──────────────────┐
         ▼                  ▼                  ▼                  ▼
  [ codegen_cpp.py ] [ codegen_py.py ] [ codegen_tools.py ] [ report_html.py ]
    #pragma once       Standalone        ReClass.NET XML      Dark-Mode HTML5
    C++20 + static_    Python 3.10+      Cheat Engine .CT     Cache-Line Map
    assert + Reader    Unpacker          (Bottom-Up Offsets)  + Inline SVG
```

---

## 3-Gate Verification Oracle (`0–100`)

Every synthesized or hand-authored `SdkSpec` is scored and gated by `verify_sdk()`:

1. **Gate 1 — Strict x86-64 ABI Layout & Anti-Triviality:**
   - Contiguous, zero-gap, zero-overlap fields from `0` to `sizeof`.
   - Natural field alignment (`offset % alignment == 0`).
   - Exact struct alignment (`max(field.alignment)`) and tail-padding multiple (`size % alignment == 0`).
   - Non-padding semantic density `>= 15%`.
2. **Gate 2 — Multi-Snapshot Replay & RTTI VPtr Proof:**
   - Every `PointerChain` dereferences cleanly across all snapshots and matches `expected_values`.
   - Every `VPTR` field resolves to `image_base + vtable_rva`; every `STD_VECTOR` satisfies `begin <= end <= cap`.
3. **Gate 3 — 100-Trial Deterministic ASLR Relocation & Struct-Drift Fuzzer:**
   - Relocates `image_base` and heap regions across `100` deterministic trials (`seed = 0x5746049E`) and verifies `100%` chain + `VPTR` survival (`aslr_pass_rate == 1.0`).
   - Injects `8`/`16`-byte field shifts at `+0x08` across `50` trials and verifies differential leaf re-anchoring (`drift_reanchor_rate`).
   - Score formula: `50 * aslr_pass_rate + 25 * min(1, semantic_density / 0.50) + 25 * drift_reanchor_rate` (requires `aslr_pass_rate == 1.0` and `layout_score >= 80.0`).

---

## CLI Quickstart

Generate the sample PE32+/ELF64 binaries and 3 heap snapshots (`examples/samples/`):

```bash
python examples/generate_demo_dumps.py
```

1. **Harvest RTTI & VTables from PE32+ or ELF64:**

```bash
structforge rtti examples/samples/game_rtti.dll
structforge rtti examples/samples/engine_rtti.elf --json
```

2. **Synthesize an `SdkSpec` via differential watches (`root_va = 0x140005000`):**

```bash
structforge synthesize \
  --snapshots examples/samples/snap_0.sfdump examples/samples/snap_1.sfdump examples/samples/snap_2.sfdump \
  --binary examples/samples/game_rtti.dll \
  --module game_rtti.dll \
  --name GameSdk \
  --watch health:f32:100.0,72.5,0.0 \
  --watch ammo:i32:30,29,0 \
  --watch is_alive:bool:true,true,false \
  -o game_sdk.json \
  --report report.html
```

3. **Verify through the 3-Gate Oracle:**

```bash
structforge verify game_sdk.json \
  --snapshots examples/samples/snap_0.sfdump examples/samples/snap_1.sfdump examples/samples/snap_2.sfdump \
  --binary examples/samples/game_rtti.dll
```

4. **Compile to C++20, Python, ReClass.NET, or Cheat Engine:**

```bash
structforge compile game_sdk.json --lang cpp -o game_sdk.hpp
structforge compile game_sdk.json --lang python -o game_sdk.py
structforge compile game_sdk.json --lang reclass -o game_sdk.rcnet
structforge compile game_sdk.json --lang ce -o game_sdk.CT
```

5. **Generate Standalone Dark-Mode HTML5 Inspector:**

```bash
structforge inspect game_sdk.json \
  --snapshots examples/samples/snap_0.sfdump examples/samples/snap_1.sfdump examples/samples/snap_2.sfdump \
  -o report.html
```

---

## Development & Tests

```bash
python -m pip install -e ".[dev]"
ruff format --check .
ruff check .
pytest -v
```
