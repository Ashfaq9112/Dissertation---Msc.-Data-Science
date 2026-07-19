# gamma_bench — common spine for the Γ-field compression comparison

A dependency-light skeleton implementing the code-level specification from the
experimental design. It exists to make the crux ablation a *controlled*
experiment: every allocation method consumes the same `LayerStats` and feeds
the same `Quantizer`, so the only thing that varies is the `AllocationMap`.

## Layout
    gamma_bench/
      core/structures.py     # the spine: LayerStats, AllocationMap, Quantizer, GeometryProbe
      adapters/adapters.py   # per-family hooks: gamma, ladder L0-L4, rotation, reused PTQ/VQ
      results/schema.py      # the common output shell: RunRecord + JSONL store
      results/analysis.py    # statistics + figure shells keyed to H1-H6 and T1-T3
    run_demo.py              # end-to-end smoke test on synthetic data

## The spine (4 objects)
- **LayerStats** — single source of second-order info (Σ, and an HVP action for
  the L4 near-true-Hessian rung). Backs *every* allocator.
- **AllocationMap** — layer→bits; the ONLY object an allocation method varies.
- **Quantizer** — fixed realization of an AllocationMap onto weights.
- **GeometryProbe** — fixed CKA and k-NN-overlap measurement.

## Per-family hooks (what each method must supply)
| Family | Strategy | Hooks |
|---|---|---|
| Γ-field | reuse Γ_prac | `allocate` via SVD-free λ_geo estimators; `pre_transform`=identity |
| Ladder L0–L4 | reimplement (thin) | one `allocate` with 5 sensitivity backends over LayerStats |
| Rotation (QuaRot/SpinQuant) | reuse Q | `pre_transform` applies published Q; inner allocator runs on rotated weights |
| Reused PTQ/VQ (GPTQ/AWQ/APTQ/AQLM) | reuse | `extract_alloc` (effective bits) + `run_native` (own headline) |

Each baseline reports TWO paths: native (own repo, honest headline) and
controlled (its AllocationMap → shared Quantizer). Reporting both pre-empts the
"did you weaken the baselines?" objection.

## Output shell → statistics → figures
Every run writes one `RunRecord` (uniform schema) to a JSONL store. Analysis
builds only from records:
- `crux_ladder_table` / `plot_ladder_geometry` → H1/H2/H3
- `rotation_interaction_table` → H4/H5/H6
- capacity fields → T1; roofline fields + `plot_roofline` → T2;
  `geometry_quality_regression` / `plot_geometry_quality` → T3

## Run the smoke test
    pip install pandas matplotlib
    python run_demo.py
