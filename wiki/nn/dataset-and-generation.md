# Neural-Network Dataset and Generation

This page owns the binary dataset contract and source-visible generation workflow. Authority is `src/pyghl/nn_c2p/nn_c2p_generate_dataset.py`, with readers in `src/pyghl/_nn_dataset.py`, `src/pyghl/nn.py`, and `src/pyghl/nn_c2p/nn_c2p_test.py`; no dataset was generated or numerically validated for this knowledge base.

## Preconditions and Cost

`generate_dataset()` imports `pyghl`, initializes GRHayL parameters, opens a tabulated EOS with fixed atmosphere/range/precision arguments, and constructs a flat metric. It therefore needs usable compiled bindings and a readable EOS table. It performs `n_pts**5` records and rejects `n_pts < 2`; default `n_pts=16` means 1,048,576 records before any extra scan work. `x_best_correction` additionally evaluates `scan_points` candidates per record and rejects `scan_points < 1`.

Generation is CPU/EOS intensive, writes a user-selected path, and opens that path with `"wb"`: an existing file is truncated without an overwrite prompt. Direct generation has no temporary-file transaction and leaves a partial file after failure.

## Binary Contract

Producer writes `struct.pack("<QQQ", 4, 16, n_blocks)`, followed by one `struct.Struct("<16f")` record per block. Thus project contract is:

| Region | Project format | Project meaning |
| --- | --- | --- |
| Bytes 0–23 | `<QQQ` | 4-byte float size, 16 floats per block, `n_pts**5` blocks |
| Remaining records | `<16f` | 16 little-endian binary32 values in field order below |

`<` meaning, standard sizes, lack of implicit alignment, `Q` size, and `f` binary32 representation come from Python `struct`; header values and field meanings come from project source.

| Index | Field | Producer meaning | Training use |
| ---: | --- | --- | --- |
| 0 | `rho` | Rest-mass density | Not selected |
| 1 | `temp` | Temperature | Not selected |
| 2 | `ye` | Electron fraction | Not selected |
| 3 | `W` | Lorentz factor | Not selected |
| 4 | `log_PmagoP` | `log10` magnetic-to-fluid-pressure ratio | Not selected |
| 5 | `vx` | Velocity x component | Not selected |
| 6 | `vy` | Velocity y component | Not selected |
| 7 | `vz` | Velocity z component | Not selected |
| 8 | `Bx` | Magnetic-field x component | Not selected |
| 9 | `By` | Magnetic-field y component | Not selected |
| 10 | `Bz` | Magnetic-field z component | Not selected |
| 11 | `q` | `tau/D` | Input column 0 |
| 12 | `r` | `S_squared/D**2` | Input column 1 |
| 13 | `s` | `B_squared/D` | Input column 2 |
| 14 | `t` | `BdotS/D**1.5` | Input column 3 |
| 15 | `x` | Selected correction target | Target column 0 |

`_nn_dataset.read_training_dataset()` accepts header float widths of 4 or 8 bytes, verifies exact payload byte count, requires at least 16 columns, selects `(11,12,13,14,15)`, and converts the result to `float32`. It accepts both target-mode spellings but selects column 15 for either. By contrast, `nn.iter_dataset_points()` requires exactly 4-byte floats and exactly 16 columns, reads exactly `n_blocks`, and does not check trailing bytes. `nn_c2p_test.py` reads only `n_blocks` from the header before seeking by `24 + offset * calcsize("<16f")`; it does not independently validate the first two header values.

## Generation Modes and Targets

`dataset_type` is exactly `train` or `test`.

- `train` uses a five-axis grid over log density, log temperature, electron fraction, Lorentz factor, and log magnetic pressure ratio. `random.Random(0)` still supplies field/velocity orientation for each record.
- `test` uses `random.Random(42)` to sample all five state axes plus orientation. These are local deterministic pseudo-random streams in source; they do not establish physics coverage or cross-version numerical identity.
- `x_correction` stores `x_exact = h * W` and configures `calc_prim_guess=True` with no GRHayL main recovery routine.
- `x_best_correction` configures the chosen C2P method, sets `calc_prim_guess=False`, and scans the closed interval from `1+q-s` to `2+2q-s`. Successful recoveries rank before failures, then by iteration count and mean conservative reconstruction error. A primitive-reconstruction exception skips that candidate. A recovery `GRHayLError` receives the fixed key `(failure, 1000000000, inf)`, so if no recovery succeeds the first candidate that reached recovery is selected; if every reconstruction raises, the lower bound is returned. A nonfinite or nonpositive interval width also returns the lower bound. Source does not mark these fallback labels as failures.

Before writing, producer sums all 16 values and rejects a nonfinite result. That check proves only finite Python values at that point; packing to binary32 can still fail or change precision, and no numerical-validity claim follows.

## Validation, Outputs, and Cleanup

Default output is `nn_training_dataset.bin` or `nn_test_dataset.bin`; explicit `output` replaces it. Progress and observed `q,r,s,t,x` ranges go to standard output. Reader failures distinguish incomplete header, unsupported float size, exact byte-size mismatch, insufficient columns, unsupported target mode, and mid-block EOF.

Installed training orchestration creates a temporary pathname with `mkstemp`, closes and unlinks the empty file, generates at that path, then deletes the generated dataset in `finally`. This cleanup applies only to its auto-generated dataset. Direct generation and explicit datasets remain user-owned.

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Binary dataset | `generate_dataset()` | `_nn_dataset.read_training_dataset()`, `nn.iter_dataset_points()`, evaluator | Default current directory or explicit path | User-owned/generated | `wb` truncates existing path | Input validation; direct failure may leave partial file | `implemented`, producer/reader source trace |
| Auto-generated training dataset | `nn_c2p_train.main()` then generator | `ghl.nn.train_on_dataset()` | OS temp directory | Temporary/user process | Fresh temp name; empty reservation is unlinked before generation | `finally` unlinks generated path, including after training error | `implemented` only |
| Evaluation text | `nn_c2p_test.main()` | User analysis | Default `nn_test*.asc` or explicit path | User-owned/generated | Text `w` truncates existing output | No transactional cleanup | `implemented` only |

## Evidence and Gaps

- Producer and three reader paths were traced at source level.
- No root test directly covers this schema, producer/reader agreement, scan ranking, generation cleanup, or numerical behavior.
- No generation command was run. EOS availability, generated ranges, record count, native calls, and scientific validity remain unproved.

## Change Impact

Changing header or field order requires synchronized review of all reader paths,
training selection constants, examples, evaluator seeks, and any external
consumers. Changing target calculation requires training target decoding,
exported representations, parent-pinned C behavior, and comparison tests. Cost
or output-path changes require [CLI workflows](../cli/command-workflows.md)
cleanup review.

## External Ground Truth

- [Python `struct` byte order, size, alignment, and format characters](https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment) defines `<` as little-endian standard-size/no-alignment layout and `f`/`Q` representations used here.
