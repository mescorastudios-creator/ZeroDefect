# ZeroDefect — Production simulator & virtual camera

Public defect datasets have real images but **no production metadata**. The simulator creates that metadata for a virtual injection-moulding plant and binds every part to a real dataset image of the matching defect class. **The metadata is simulated. The images and defects are real.** Every output says so.

## Run it

```bash
uv run python -m ml.datasets demo                  # or download + convert Real-IAD
uv run python -m simulator run --images demo       # 14-day history -> data/sim/demo/
uv run python -m simulator run --images realiad --days 28
uv run python -m simulator run --images none       # metadata only, no image binding
```

A 14-day run produces about 450,000 parts in under 10 seconds.

## Plant

Everything is set in [`configs/plant.yaml`](../configs/plant.yaml). Invalid references (unknown machine, supplier, class, parameter …) fail at load time.

| | |
|---|---|
| Lines / machines | 3 lines × 2 injection-moulding machines (`IMM-01…06`), 2-cavity moulds. Each machine makes one part type (the Real-IAD polymer classes; `plastic_nut` runs on two machines, so identical parts can be compared). |
| Shifts | A 06–14, B 14–22, C 22–06. Night-shift parts after midnight belong to the previous `shift_date`. |
| Operators | 9, in 3 teams, one operator per line per shift. Teams rotate shifts weekly and operators rotate lines daily. This keeps operator, line and shift effects statistically separable. |
| Resin | 4 suppliers. Each machine switches supplier every 8 h according to its `resin_mix` and uses that supplier's current lot (`SUP-C-L005` = SUP-C's 5th lot). Several machines can share a lot. Lots have hidden moisture/contamination factors; `lots.csv` holds the incoming-inspection moisture measurement. |
| Process | Per shot: melt temperature, injection pressure, holding time, cooling time, mould temperature, cycle time. Each is setpoint + slow AR(1) drift + shot-to-shot noise. Shot times follow the actual cycle times. |
| Stoppages | Random unplanned stops (≈1.2 per machine-day), plus maintenance stops that end drift faults. |

## Defect model

For each part and class:

```
p(class) = base_rate × exp(Σ coef × (parameter − setpoint) / tolerance) × lot_factor^exponent × fault multipliers
```

The coefficients encode injection-moulding knowledge:

| Defect | Rises with |
|---|---|
| missing_material (short shot) | low melt temperature, low injection pressure, low mould temperature |
| pit_void (sink / void) | short holding time, short cooling time, high melt temperature, wet resin |
| deformation (warpage) | short cooling time, high mould temperature |
| discoloration (burn) | high melt temperature, long cycle (residence time), wet resin |
| crack | low mould temperature, high injection pressure |
| contamination | contaminated resin lot |

One outcome (good or one class) is drawn per part. When images are bound, classes with no image for that part type are switched off, with a warning. The recorded ground truth therefore always matches the image.

## Planted faults (ground truth for root-cause evaluation)

| Id | Fault | Scope | Effect |
|---|---|---|---|
| F1 | Failing heater band: mould temperature drifts −2.5 °C/day | IMM-03, days 3.4–8.6, then 3 h maintenance | short shots (and cracks) rise ~3× on average, ~5× at the end |
| F2 | Contaminated resin lot | lot `SUP-C-L005`, on any machine | contamination ×8 |
| F3 | Worn mould cavity | IMM-05 cavity 2, from day 6 | deformation ×3.5, scratch ×1.8 |
| F4 | Rough part handling | operator OP-08 | scratch ×2.2 |
| — | Wet resin from one supplier | SUP-D lots (moisture 1.6×) | more voids |

Each run's `ground_truth.json` lists the faults with their absolute times and the observed lift. The M5 root-cause analysis must find them **without** reading that file. The M2/M7 report then scores it by top-1/top-3 hit rate against the file. On the default config, a demo run shows lifts of 3.1× (F1), 7.4× (F2), 2.3× (F3) and 2.2× (F4).

Fault times are in days from the simulation start. A live run started "now" therefore still plays them out.

## Outputs (`data/sim/<name>/`)

| File | Content |
|---|---|
| `parts.parquet` | One row per part: `part_id` (`ZD-IMM03-0012345-2` = machine, shot, cavity), `timestamp`, `shift_date`, `shift`, `line_id`, `machine_id`, `mould_id`, `cavity`, `shot_no`, `part_type`, `material`, `operator_id`, `supplier_id`, `resin_lot`, the 6 process parameters, `true_label`, `true_class`, `image_sample_id` |
| `lots.csv` | Resin lots: supplier, received time, measured moisture % |
| `stoppages.csv` | Machine stops with reason |
| `ground_truth.json` | Planted faults, supplier quality deviations, observed lifts |
| `plant.json` | The exact configuration used |

Generation runs one machine-day at a time, with a random generator seeded by (seed, machine, day) and state carried between days. A batch history and a live stream therefore produce identical records.

## Virtual camera

`simulator/camera.py` turns any source into timestamped `Capture`s: one part with 1–N views, or one video frame. `paced()` releases them in real time or faster. This is the input of the M1 inspection pipeline.

```bash
# simulated parts with their 5 real views, 60x real time, contact sheets for the first 20
uv run python -m simulator.camera --source sim --images demo --speed 60 --limit 20 --out outputs/preview
# live timeline starting now, real time
uv run python -m simulator.camera --source sim --images realiad --live --limit 0
# a folder of images (Real-IAD style C1..C5 files are grouped per part), a video, a webcam or an RTSP stream
uv run python -m simulator.camera --source folder:data/raw/mvtec_ad/screw/test --interval 0.5
uv run python -m simulator.camera --source video:line.mp4
uv run python -m simulator.camera --source video:0
uv run python -m simulator.camera --source video:rtsp://camera.local/stream
```

Simulated parts use the dataset's **test** split by default, so a model trained on the train split never sees them.
