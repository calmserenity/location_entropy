"""Build presentations/location_entropy_deck.html from analysis notebook + assets.

Run from repo root:
    python scripts/build_interview_deck.py

Extracts Matplotlib PNGs and Plotly figure JSON from the executed notebook,
then regenerates the interview deck HTML.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_pngs(nb: dict) -> list[bytes]:
    images: list[bytes] = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                images.append(base64.b64decode(data["image/png"]))
    return images


def extract_plotly_figs(nb: dict) -> list[dict]:
    figs: list[dict] = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            fig = data.get("application/vnd.plotly.v1+json")
            if fig:
                figs.append(fig)
    return figs


def b64_json(obj: dict) -> str:
    return base64.b64encode(json.dumps(obj, separators=(",", ":")).encode()).decode("ascii")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Location entropy — analysis &amp; validation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700;800&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    html, body { height: 100%; overflow-x: hidden; margin: 0; }
    html { scroll-snap-type: y mandatory; scroll-behavior: smooth; }
    .slide {
      width: 100vw; height: 100vh; height: 100dvh; overflow: hidden;
      scroll-snap-align: start; display: flex; flex-direction: column; position: relative;
    }
    .slide-content {
      flex: 1; display: flex; flex-direction: column; justify-content: center;
      max-height: 100%; overflow: hidden; padding: var(--slide-padding);
    }
    :root {
      --title-size: clamp(1.5rem, 4.5vw, 3.25rem);
      --h2-size: clamp(1.15rem, 3vw, 2.1rem);
      --h3-size: clamp(0.95rem, 2.2vw, 1.4rem);
      --body-size: clamp(0.72rem, 1.35vw, 1.05rem);
      --small-size: clamp(0.62rem, 0.95vw, 0.82rem);
      --slide-padding: clamp(0.75rem, 3.5vw, 3rem);
      --content-gap: clamp(0.4rem, 1.6vw, 1.35rem);
      --element-gap: clamp(0.22rem, 0.9vw, 0.85rem);
      --swiss-red: #e30613;
      --swiss-ink: #0a0a0a;
      --swiss-paper: #f6f6f4;
      --swiss-rule: #ccc;
    }
    body {
      font-family: "Nunito", system-ui, sans-serif; background: var(--swiss-paper); color: var(--swiss-ink);
    }
    .deck-grid-decor {
      pointer-events: none; position: absolute; inset: 0;
      background-image: linear-gradient(to right, var(--swiss-rule) 1px, transparent 1px),
        linear-gradient(to bottom, var(--swiss-rule) 1px, transparent 1px);
      background-size: 56px 56px; opacity: 0.2;
    }
    .slide-inner { position: relative; z-index: 1; width: 100%; max-width: min(94vw, 1140px); margin: 0 auto; }
    h1 {
      font-family: "Archivo", sans-serif; font-size: var(--title-size); font-weight: 800;
      letter-spacing: -0.03em; line-height: 1.06; margin: 0 0 var(--element-gap);
    }
    h2 {
      font-family: "Archivo", sans-serif; font-size: var(--h2-size); font-weight: 700;
      letter-spacing: -0.02em; margin: 0 0 var(--content-gap);
    }
    .kicker {
      font-family: "Archivo", sans-serif; font-size: var(--small-size); font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.11em; color: var(--swiss-red);
      margin-bottom: var(--element-gap);
    }
    .subtitle { font-size: var(--body-size); max-width: 40em; opacity: 0.88; line-height: 1.45; }
    .rule { height: 3px; width: clamp(2.5rem, 10vw, 5rem); background: var(--swiss-red); margin: var(--content-gap) 0; }
    .bullet-list {
      list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column;
      gap: clamp(0.35rem, 0.9vh, 0.85rem);
    }
    .bullet-list li {
      font-size: var(--body-size); line-height: 1.38; padding-left: 1.05em; position: relative;
    }
    .bullet-list li::before {
      content: ""; position: absolute; left: 0; top: 0.52em; width: 0.32em; height: 0.32em; background: var(--swiss-red);
    }
    .stat-row {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: var(--content-gap); margin-top: var(--content-gap);
    }
    .stat {
      border: 1px solid var(--swiss-ink); padding: clamp(0.45rem, 1.2vw, 0.85rem); background: #fff;
    }
    .stat strong { font-family: "Archivo", sans-serif; display: block; font-size: var(--h3-size); font-weight: 800; }
    .stat span { font-size: var(--small-size); opacity: 0.78; }
    .fig-wrap {
      margin-top: var(--content-gap); flex: 1; min-height: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .fig-wrap img {
      max-width: 100%; max-height: min(52vh, 520px); object-fit: contain; border: 1px solid #bbb; background: #fff;
    }
    .fig-caption { font-size: var(--small-size); opacity: 0.85; margin-top: var(--element-gap); max-width: 52em; line-height: 1.4; }
    .plot-host { width: 100%; height: min(58vh, 560px); min-height: 220px; }
    .plot-title { font-family: "Archivo", sans-serif; font-size: var(--h3-size); margin: 0 0 var(--element-gap); }
    .data-table { width: 100%; border-collapse: collapse; font-size: var(--small-size); margin-top: var(--content-gap); }
    .data-table th, .data-table td { border: 1px solid var(--swiss-ink); padding: 0.35rem 0.5rem; text-align: left; }
    .data-table th { background: var(--swiss-ink); color: #fff; font-family: "Archivo", sans-serif; }
    .deck-progress {
      position: fixed; bottom: clamp(0.45rem, 1.8vw, 0.85rem); right: clamp(0.45rem, 1.8vw, 0.85rem);
      z-index: 20; font-family: "Archivo", sans-serif; font-size: var(--small-size); font-weight: 700;
      background: var(--swiss-ink); color: #fff; padding: 0.3rem 0.65rem;
    }
    .keyboard-hint {
      position: fixed; bottom: clamp(0.45rem, 1.8vw, 0.85rem); left: clamp(0.45rem, 1.8vw, 0.85rem);
      z-index: 20; font-size: calc(var(--small-size) * 0.92); opacity: 0.55; max-width: min(88vw, 300px);
    }
    .reveal { opacity: 0; transform: translateY(10px); transition: opacity 0.4s ease, transform 0.4s ease; }
    .slide.is-active .reveal { opacity: 1; transform: translateY(0); }
    .slide.is-active .reveal:nth-child(1) { transition-delay: 0.04s; }
    .slide.is-active .reveal:nth-child(2) { transition-delay: 0.08s; }
    .slide.is-active .reveal:nth-child(3) { transition-delay: 0.12s; }
    .slide.is-active .reveal:nth-child(4) { transition-delay: 0.16s; }
    .slide.is-active .reveal:nth-child(5) { transition-delay: 0.2s; }
    @media (max-height: 640px) {
      :root { --slide-padding: clamp(0.5rem, 2.5vw, 1.25rem); --body-size: clamp(0.68rem, 1.15vw, 0.92rem); }
      .fig-wrap img { max-height: min(44vh, 420px); }
      .plot-host { height: min(50vh, 460px); }
      .keyboard-hint { display: none; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.15s !important; }
      html { scroll-behavior: auto; }
    }
  </style>
</head>
<body>
  <main id="deck" aria-label="Presentation slides">
__SLIDES__
  </main>
  <div class="deck-progress" id="progress" aria-live="polite">1 / __N_SLIDES__</div>
  <p class="keyboard-hint">Arrows · Space · PgUp/Dn · wheel · swipe</p>
  <script type="text/plain" id="plotly-b64-0">__PLOTLY0__</script>
  <script type="text/plain" id="plotly-b64-1">__PLOTLY1__</script>
  <script type="text/plain" id="plotly-b64-2">__PLOTLY2__</script>
  <script>
    (function () {
      "use strict";
      var slides = Array.from(document.querySelectorAll("#deck .slide"));
      var progressEl = document.getElementById("progress");
      var total = slides.length;
      var index = 0;
      var wheelLock = false;
      var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      function scrollBehavior() { return reduceMotion ? "auto" : "smooth"; }
      function goTo(i) {
        index = Math.max(0, Math.min(total - 1, i));
        slides[index].scrollIntoView({ block: "start", behavior: scrollBehavior() });
        progressEl.textContent = (index + 1) + " / " + total;
      }
      function next() { goTo(index + 1); }
      function prev() { goTo(index - 1); }

      document.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight" || e.key === "ArrowDown" || e.key === " " || e.key === "PageDown") { e.preventDefault(); next(); }
        else if (e.key === "ArrowLeft" || e.key === "ArrowUp" || e.key === "PageUp") { e.preventDefault(); prev(); }
        else if (e.key === "Home") { e.preventDefault(); goTo(0); }
        else if (e.key === "End") { e.preventDefault(); goTo(total - 1); }
      });

      window.addEventListener("wheel", function (e) {
        if (wheelLock) return;
        if (Math.abs(e.deltaY) < 18) return;
        wheelLock = true;
        if (e.deltaY > 0) next(); else prev();
        setTimeout(function () { wheelLock = false; }, 520);
      }, { passive: true });

      var touchStartY = null;
      document.addEventListener("touchstart", function (e) { touchStartY = e.changedTouches[0].screenY; }, { passive: true });
      document.addEventListener("touchend", function (e) {
        if (touchStartY == null) return;
        var dy = touchStartY - e.changedTouches[0].screenY;
        touchStartY = null;
        if (Math.abs(dy) < 36) return;
        if (dy > 0) next(); else prev();
      }, { passive: true });

      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          var slide = entry.target;
          if (!entry.isIntersecting) { slide.classList.remove("is-active"); return; }
          if (entry.intersectionRatio > 0.5) {
            slides.forEach(function (s) { s.classList.remove("is-active"); });
            slide.classList.add("is-active");
            index = slides.indexOf(slide);
            progressEl.textContent = index + 1 + " / " + total;
          }
        });
      }, { threshold: [0, 0.51, 1] });
      slides.forEach(function (s) { observer.observe(s); });

      function decodeFig(id) {
        var el = document.getElementById(id);
        if (!el || !el.textContent.trim()) return null;
        return JSON.parse(atob(el.textContent.trim()));
      }

      var plotlyConfig = { responsive: true, displayModeBar: true, displaylogo: false };
      [0, 1, 2].forEach(function (i) {
        var host = document.getElementById("plotly-host-" + i);
        if (!host) return;
        var fig = decodeFig("plotly-b64-" + i);
        if (!fig || typeof Plotly === "undefined") return;
        var layout = Object.assign({}, fig.layout || {}, {
          margin: { l: 40, r: 25, t: 36, b: 32 },
          paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
        });
        Plotly.newPlot(host, fig.data || [], layout, plotlyConfig);
      });
    })();
  </script>
</body>
</html>
"""


def slide_block(body: str, slide_id: int, first: bool = False) -> str:
    active = " is-active" if first else ""
    return f"""    <section class="slide{active}" id="slide-{slide_id}" aria-labelledby="slide-h-{slide_id}">
      <div class="deck-grid-decor" aria-hidden="true"></div>
      <div class="slide-content">
        <div class="slide-inner">
{body}
        </div>
      </div>
    </section>
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nb_path = root / "location_entropy_analysis.ipynb"
    out_path = root / "presentations" / "location_entropy_deck.html"
    assets_dir = root / "presentations" / "assets"

    nb = load_notebook(nb_path)
    pngs = extract_pngs(nb)
    figs = extract_plotly_figs(nb)

    if len(pngs) < 3:
        raise SystemExit(f"Expected 3 PNG outputs in notebook, found {len(pngs)}. Re-run Step 6 in the notebook.")
    if len(figs) < 3:
        raise SystemExit(f"Expected 3 Plotly outputs in notebook, found {len(figs)}. Re-run Step 6 in the notebook.")

    assets_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "fig_entropy_distribution.png",
        "fig_entropy_vs_locations.png",
        "fig_representative_density.png",
    ]
    for p, name in zip(pngs, names, strict=True):
        (assets_dir / name).write_bytes(p)

    plotly_b64 = [b64_json(figs[i]) for i in range(3)]
    plotly_labels = [
        "Low entropy example — single-day 3D trajectory (lon, lat, hour)",
        "Median entropy example — single-day 3D trajectory",
        "High entropy example — single-day 3D trajectory",
    ]

    parts: list[str] = []
    sid = 0

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Cabspotting · Location entropy</p>
          <h1 class="reveal" id="slide-h-{sid}">Analysis &amp; validation results</h1>
          <div class="rule reveal" aria-hidden="true"></div>
          <p class="subtitle reveal">Time-weighted mobility entropy on a discrete grid — dataset checks, quantitative results, and visual summaries for 536 vehicles.</p>""",
            sid,
            first=True,
        )
    )
    sid += 1

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Overview</p>
          <h2 class="reveal" id="slide-h-{sid}">Goal, input, and scoring method</h2>
          <ul class="bullet-list">
            <li class="reveal"><strong>Goal</strong> — Score how predictable each vehicle is: habitual drivers (few areas, repeatedly) vs. exploratory drivers (time scattered across many areas).</li>
            <li class="reveal"><strong>Input</strong> — 536 vehicles, ~11M GPS pings with timestamps. We use <em>elapsed time</em> between pings as a proxy for “time spent,” not raw ping counts.</li>
            <li class="reveal"><strong>Locations</strong> — Group nearby coordinates into ~500 m map squares (<strong>6,161</strong> unique squares in this dataset). This prevents “every ping is a new place.”</li>
            <li class="reveal"><strong>Score</strong> — Entropy measures how <em>evenly</em> time is split across squares. Low = concentrated in a few squares; high = spread thinly across many. We also record the % of time in the single busiest square.</li>
            <li class="reveal"><strong>Data quality</strong> — Gaps longer than 3 hours are treated as “unknown,” not as staying in the last seen square. This prevents phantom dwell time.</li>
          </ul>""",
            sid,
        )
    )
    sid += 1

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Method</p>
          <h2 class="reveal" id="slide-h-{sid}">How we calculated entropy</h2>
          <ul class="bullet-list">
            <li class="reveal"><strong>Step 1 — Time between pings.</strong> For each vehicle, we looked at consecutive GPS points and measured the minutes between them. This time is credited to the starting location as "dwell time."</li>
            <li class="reveal"><strong>Step 2 — Group into squares.</strong> Raw lat/lon coordinates are nearly unique every second, so we bucket nearby points into ~500 meter map squares. Nearby pings count as the same place.</li>
            <li class="reveal"><strong>Step 3 — Build a time budget.</strong> For each vehicle, we sum all dwell minutes per square, then convert to percentages (e.g., "Square A: 15%, Square B: 8%..."). This forms a probability distribution.</li>
            <li class="reveal"><strong>Step 4 — Score with entropy.</strong> We apply the standard entropy formula: <em>H = −Σ p × log₂(p)</em>. A driver with 90% in one square scores near 0; a driver with 10% in ten squares scores higher.</li>
            <li class="reveal"><strong>Step 5 — Quality filter.</strong> Gaps over 3 hours or backward timestamps are dropped. This prevents phantom dwell from missing data.</li>
          </ul>""",
            sid,
        )
    )
    sid += 1

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Test results</p>
          <h2 class="reveal" id="slide-h-{sid}">Pipeline validation</h2>
          <div class="stat-row">
            <div class="stat reveal"><strong>536</strong><span>files / users</span></div>
            <div class="stat reveal"><strong>11.2M</strong><span>rows loaded, 0 skipped</span></div>
            <div class="stat reveal"><strong>6,161</strong><span>unique grid cells</span></div>
            <div class="stat reveal"><strong>783.3M s</strong><span>total observed time</span></div>
            <div class="stat reveal"><strong>11.21M</strong><span>transitions used</span></div>
            <div class="stat reveal"><strong>8,464</strong><span>transitions skipped</span></div>
          </div>
          <p class="fig-caption reveal">Skipped transitions: non-positive Δ<em>t</em> or gaps above the max-gap threshold — avoids inflating inferred dwell.</p>""",
            sid,
        )
    )
    sid += 1

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Analysis outputs</p>
          <h2 class="reveal" id="slide-h-{sid}">Entropy summary &amp; export</h2>
          <div class="stat-row">
            <div class="stat reveal"><strong>7.07</strong><span>mean <em>H</em> (bits)</span></div>
            <div class="stat reveal"><strong>0.56</strong><span>min <em>H</em></span></div>
            <div class="stat reveal"><strong>7.85</strong><span>max <em>H</em></span></div>
            <div class="stat reveal"><strong>536</strong><span>rows in ranked CSV</span></div>
          </div>
          <p class="fig-caption reveal">Deliverable: <code>outputs/stepwise_location_entropy_results.csv</code>. Highest-entropy example — user <code>eapceou</code>: 7.85 bits, 796 squares visited, single busiest square only 2.6% of total time. This pattern shows time is spread thinly across many places, not just that the vehicle visited many places.</p>""",
            sid,
        )
    )
    sid += 1

    rel_img = "assets/fig_entropy_distribution.png"
    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Figure</p>
          <h2 class="reveal" id="slide-h-{sid}">Entropy distribution across users</h2>
          <div class="fig-wrap reveal">
            <img src="{rel_img}" alt="Histogram of location entropy with mean and median reference lines" width="1000" height="600" decoding="async">
          </div>
          <p class="fig-caption reveal">Distribution of per-user <em>H</em> (bits) with mean and median markers — shows spread of routine vs diffuse mobility in the sample.</p>""",
            sid,
        )
    )
    sid += 1

    rel_img2 = "assets/fig_entropy_vs_locations.png"
    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Figure</p>
          <h2 class="reveal" id="slide-h-{sid}">Entropy vs number of locations</h2>
          <div class="fig-wrap reveal">
            <img src="{rel_img2}" alt="Scatter plot of entropy versus number of observed locations" width="1000" height="600" decoding="async">
          </div>
          <p class="fig-caption reveal">Each point = one user. Color = top-location share; size ∝ observed time. Annotated points = low / median / high entropy examples used later.</p>""",
            sid,
        )
    )
    sid += 1

    rel_img3 = "assets/fig_representative_density.png"
    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Figure</p>
          <h2 class="reveal" id="slide-h-{sid}">Representative GPS density maps</h2>
          <div class="fig-wrap reveal">
            <img src="{rel_img3}" alt="Hexbin maps of GPS point density for three representative users" width="1800" height="500" decoding="async">
          </div>
          <p class="fig-caption reveal">Hexbins of downsampled pings (density proxy, not exact dwell). Compares spatial concentration for low vs median vs high entropy exemplars.</p>""",
            sid,
        )
    )
    sid += 1

    for pi, label in enumerate(plotly_labels):
        parts.append(
            slide_block(
                f"""          <p class="kicker reveal">Interactive figure</p>
          <h2 class="plot-title reveal" id="slide-h-{sid}">{label}</h2>
          <p class="fig-caption reveal">Longitude and latitude in the plane; height encodes hour of day (marker color matches). Use the controls to rotate and zoom.</p>
          <div id="plotly-host-{pi}" class="plot-host reveal" role="img" aria-label="3D trajectory plot"></div>""",
                sid,
            )
        )
        sid += 1

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Segment summary</p>
          <h2 class="reveal" id="slide-h-{sid}">Low / medium / high entropy</h2>
          <table class="data-table reveal" aria-label="Average metrics by entropy segment">
            <thead><tr><th>Segment</th><th>Users</th><th>Avg <em>H</em></th><th>Avg squares</th><th>Time in top square</th><th>Avg hours tracked</th></tr></thead>
            <tbody>
              <tr><td>Low entropy</td><td>179</td><td>6.63</td><td>623</td><td>15.8%</td><td>354</td></tr>
              <tr><td>Medium entropy</td><td>179</td><td>7.16</td><td>701</td><td>8.6%</td><td>425</td></tr>
              <tr><td>High entropy</td><td>178</td><td>7.41</td><td>744</td><td>6.6%</td><td>439</td></tr>
            </tbody>
          </table>
          <p class="fig-caption reveal">Users split into three equal groups by entropy rank. <strong>Time in top square</strong> = average share of time spent in each user’s single most-visited map square. Low-entropy users concentrate time in one area; high-entropy users spread time thinly.</p>""",
            sid,
        )
    )
    sid += 1

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">What we found</p>
          <h2 class="reveal" id="slide-h-{sid}">Three patterns in the data</h2>
          <ul class="bullet-list">
            <li class="reveal"><strong>Routine drivers work less.</strong> Low-entropy users averaged 354 hours tracked vs. 439 hours for high-entropy users. Fixed-shift drivers show up as more predictable.</li>
            <li class="reveal"><strong>It's not about counting places.</strong> High-entropy users visited only 20% more squares (744 vs. 623), but split time far more evenly. Their top square got 6.6% of time vs. 15.8% for routine drivers.</li>
            <li class="reveal"><strong>Pairing metrics catches different behaviors.</strong> Some drivers cover many squares but still camp in one favorite spot. Others roam broadly. Entropy + top-square share separates the two.</li>
          </ul>
          <p class="fig-caption reveal" style="margin-top:var(--content-gap)"><strong>Limitations:</strong> Grid size (~500 m) defines what counts as "one place"; gaps over 3 hours are treated as unknown (not stay); results reflect observed time only, not total work time.</p>""",
            sid,
        )
    )
    sid += 1

    parts.append(
        slide_block(
            f"""          <p class="kicker reveal">Thank you</p>
          <h1 class="reveal" id="slide-h-{sid}" style="font-size: var(--h2-size);">Questions?</h1>
          <div class="rule reveal" aria-hidden="true"></div>
          <p class="subtitle reveal">Detailed methodology and code: <code>location_entropy_analysis.ipynb</code></p>""",
            sid,
        )
    )
    sid += 1

    n_slides = sid
    html = (
        HTML_TEMPLATE.replace("__SLIDES__", "\n".join(parts))
        .replace("__N_SLIDES__", str(n_slides))
        .replace("__PLOTLY0__", plotly_b64[0])
        .replace("__PLOTLY1__", plotly_b64[1])
        .replace("__PLOTLY2__", plotly_b64[2])
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({n_slides} slides)")


if __name__ == "__main__":
    main()
