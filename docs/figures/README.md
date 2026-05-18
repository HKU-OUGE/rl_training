# Paper figures

Editorial-style SVG/HTML diagrams for the Rough-MoE-Teacher work.

## Files

| File | Purpose |
|---|---|
| `style.css`                       | Shared paper-figure theme (IBM Plex fonts, ink + teal accent palette) |
| `01_architecture.html`            | **Fig 1.** SplitMoE architecture (obs → encoders → GRU → gates → experts → action) |
| `02_per_rank_training.html`       | **Fig 2.** 8-rank heterogeneous terrain training topology (DDP all-reduce) |
| `03_eval_protocol.html`           | **Fig 3.** Single-pass eval protocol (terrain grid + per-tile kinematics) |
| `01_architecture.png`             | Rendered PNG, 1600×1000, transparent background |
| `02_per_rank_training.png`        | Rendered PNG |
| `03_eval_protocol.png`            | Rendered PNG |

## Aesthetic

Refined editorial / NeurIPS-Nature publication style:

- Paper-warm background `#f8f6f0`
- Deep ink primary `#1c1917` for lines and titles
- Single deep-teal accent `#0a6b6b` for the *active* policy pathway
- Single warm-amber accent `#b85c38` for the gating decision points
- IBM Plex Serif (titles + captions), IBM Plex Sans (labels), IBM Plex Mono (technical dims)
- Asymmetric figure header with kicker label, title, and a right-aligned monospace metadata strip
- Hairline rules instead of heavy borders

## Re-rendering PNGs

Headless Chrome / Chromium is the simplest exporter:

```bash
cd docs/figures
for f in 01_architecture 02_per_rank_training 03_eval_protocol; do
  google-chrome --headless --disable-gpu --no-sandbox --hide-scrollbars \
    --window-size=1600,1000 --screenshot=$f.png \
    "file://$(pwd)/$f.html"
done
```

For a *higher* DPI export (publication-ready), bump `--window-size` and re-export — the SVG scales losslessly. Example for 2× density:

```bash
google-chrome --headless --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=3200,2000 --screenshot=01_architecture@2x.png \
  --force-device-scale-factor=2 \
  "file://$(pwd)/01_architecture.html"
```

## PDF export

Same Chrome flag set, swap `--screenshot` for `--print-to-pdf`:

```bash
google-chrome --headless --disable-gpu --no-sandbox \
  --print-to-pdf=01_architecture.pdf \
  --no-pdf-header-footer \
  "file://$(pwd)/01_architecture.html"
```

LaTeX-friendly: the printed PDF crops to A4 by default; for tight-cropped figures pipe through `pdfcrop`:

```bash
pdfcrop --margins 6 01_architecture.pdf 01_architecture_cropped.pdf
```

## Editing tips

- Each HTML file is self-contained except for `style.css`. Open directly in a browser to live-edit.
- All geometry is in `viewBox="0 0 1240 700/720"` — coordinates are direct pixels in that space.
- Modify `style.css` `:root` block to change the global palette in one place.
- The pattern is plain SVG (no JS, no D3) so edits are precise and diffable.

## Referenced literature

| Paper | Figure inspiration |
|---|---|
| [MoE-Loco (IROS 2025)](https://arxiv.org/abs/2503.08564) | Fig 1 expert-bank layout · Fig 8 t-SNE coloring style |
| [Mixtral of Experts (2024)](https://arxiv.org/abs/2401.04088) | Routing arrow conventions |
| [Mod-Squad (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Mod-Squad_Designing_Mixtures_of_Experts_As_Modular_Multi-Task_Learners_CVPR_2023_paper.pdf) | Co-activation/specialization matrix conventions |
