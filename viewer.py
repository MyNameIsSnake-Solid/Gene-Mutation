"""
Build a self-contained HTML page showing, patient by patient, every
validation image with the pipeline's boxes, the chromosome each box was
assigned to, and the image-level and patient-level verdicts.

    python viewer.py                       # the default example patients
    python viewer.py 105327 10408          # specific patients (case IDs)
"""

import determinism  # noqa: F401 (import first - must precede any torch/ultralytics import)

import argparse
import base64
import glob
import html
import os
from collections import defaultdict

from PIL import Image
from ultralytics import YOLO

from assignment_solve import assign_crops, diagnose_image, get_crop_probs
from case_aggregate import NEEDS_MORE_CELLS, aggregate_by_case, case_id, direction
from config import CLASSIFIER, DETECTOR
from detect import detect
from ground_truth import load_ground_truth
from voc_to_yolo import CLASS_TO_ID, OUT_ROOT

DEFAULT_CASES = ["105327", "105101", "10321", "105027", "10408"]


def analyze(cases, device="0"):
    detector, classifier = YOLO(DETECTOR), YOLO(CLASSIFIER)
    truth = load_ground_truth()
    paths = [p for p in sorted(glob.glob(os.path.join(OUT_ROOT, "images", "val", "*")))
             if case_id(os.path.splitext(os.path.basename(p))[0]) in set(cases)]

    rows, details = [], {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        gt = truth.get(stem)
        if gt is None:
            continue
        dets = detect(detector, path, device=device)
        probs = get_crop_probs(classifier, path, dets, device=device)
        result = diagnose_image(probs)
        labels = assign_crops(probs)

        rows.append({
            "stem": stem,
            "gt_verdict": gt.verdict,
            "gt_kind": gt.kind or "normal",
            "gt_chromosomes": ";".join(gt.chromosomes),
            "pred_verdict": result.verdict,
            "pred_chromosomes": ";".join(a.chromosome for a in result.abnormalities),
            "pred_direction": direction(result.abnormalities),
        })
        details[stem] = {
            "path": path, "gt": gt, "result": result,
            "boxes": [(d["bbox"], label, extra, d["confidence"], float(p[CLASS_TO_ID[label]]))
                      for d, p, (label, extra) in zip(dets, probs, labels)],
        }
    return rows, details


def describe_truth(gt):
    total = f" ({gt.reported_total} chromosomes)" if gt.reported_total else ""
    if gt.verdict == "normal":
        return f"normal{total}"
    what = ", ".join("unspecified" if c.lower().startswith("unknow") else c
                     for c in gt.chromosomes) or "unspecified"
    return f"{gt.kind} abnormality: {what}{total}"


def patient_outcome(stems, details, agg):
    kinds = {details[s]["gt"].kind or "normal" for s in stems}
    verdicts = {agg[s]["pred_verdict"] for s in stems}
    flagged = "abnormal" in verdicts
    if "structural" in kinds:
        return "Structural (out of scope)", "neutral"
    if "numerical" in kinds:
        return ("Caught", "good") if flagged else ("Missed", "bad")
    if flagged:
        return "False alarm", "bad"
    if NEEDS_MORE_CELLS in verdicts:
        return "Needs more cells", "warn"
    return "Correctly normal", "good"


def render_image(stem, info, confirmed):
    with Image.open(info["path"]) as im:
        w, h = im.size
    mime = "image/png" if info["path"].lower().endswith(".png") else "image/jpeg"
    with open(info["path"], "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")

    result = info["result"]
    own_flags = {a.chromosome for a in result.abnormalities}
    font = max(14, w / 45)

    shapes = []
    for (x1, y1, x2, y2), label, extra, det_conf, cls_prob in info["boxes"]:
        kind = "confirmed" if label in confirmed else "own" if label in own_flags else "plain"
        classes = f"box {kind}" + (" extra" if extra else "")
        tip = (f"{label}{' (left over: extra copy)' if extra else ''} - "
               f"classifier {cls_prob:.2f}, detector {det_conf:.2f}")
        ty = y1 - 4 if y1 - 4 > font else y1 + font
        shapes.append(
            f'<g class="{classes}"><title>{html.escape(tip)}</title>'
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" height="{y2 - y1:.1f}"/>'
            f'<text x="{x1:.1f}" y="{ty:.1f}" font-size="{font:.0f}">{label}</text></g>')

    gains = [a for a in result.abnormalities if a.actual > a.expected]
    losses = [a for a in result.abnormalities if a.actual < a.expected]
    alone = ", ".join(f"{a.chromosome}: {a.actual} (expected {a.expected})" for a in gains + losses)
    missing = "".join(
        f'<li class="miss"><span class="k">Missing</span><span>{a.chromosome}: found {a.actual} of {a.expected} (no box to draw)</span></li>'
        for a in losses)

    return f"""
    <figure class="card" tabindex="0" title="Click to enlarge">
      <svg viewBox="0 0 {w} {h}" role="img" aria-label="Image {stem} with {len(info['boxes'])} detected chromosomes">
        <image href="data:{mime};base64,{data}" width="{w}" height="{h}"/>
        {''.join(shapes)}
      </svg>
      <figcaption>
        <div class="cap-head"><span class="stem">{stem}</span><span class="count">{len(info['boxes'])} boxes</span></div>
        <ul>
          <li><span class="k">Ground truth</span><span>{html.escape(describe_truth(info['gt']))}</span></li>
          <li><span class="k">This image alone</span><span>{html.escape(alone) or 'nothing flagged'}</span></li>
          {missing}
        </ul>
      </figcaption>
    </figure>"""


def render(rows, details, cases):
    agg = {r["stem"]: r for r in aggregate_by_case(rows)}
    by_case = defaultdict(list)
    for r in rows:
        by_case[case_id(r["stem"])].append(r["stem"])

    nav, sections = [], []
    for case in cases:
        stems = sorted(by_case.get(case, []))
        if not stems:
            print(f"  [skip] no validation images for patient {case}")
            continue
        outcome, tone = patient_outcome(stems, details, agg)
        verdict = agg[stems[0]]["pred_verdict"]
        confirmed = set(filter(None, agg[stems[0]]["pred_chromosomes"].split(";"))) if verdict == "abnormal" else set()
        if verdict == "abnormal":
            call = f"abnormal, confirmed across images: {', '.join(sorted(confirmed)) or 'single image'}"
        elif verdict == NEEDS_MORE_CELLS:
            call = "needs more cells (a missing chromosome seen in only one cell)"
        else:
            call = "normal"
        truths = sorted({describe_truth(details[s]["gt"]) for s in stems})

        nav.append(f'<a href="#p{case}"><span class="dot {tone}"></span>{case}<small>{outcome}</small></a>')
        sections.append(f"""
  <section id="p{case}">
    <header class="patient">
      <h2>Patient {case} <span class="badge {tone}">{outcome}</span></h2>
      <p><span class="k">Pipeline</span> {html.escape(call)}</p>
      <p><span class="k">Ground truth</span> {html.escape(' / '.join(truths))}</p>
      <p class="sub">{len(stems)} validation image{'s' if len(stems) != 1 else ''}</p>
    </header>
    <div class="grid">{''.join(render_image(s, details[s], confirmed) for s in stems)}</div>
  </section>""")

    return PAGE.replace("{NAV}", "".join(nav)).replace("{SECTIONS}", "".join(sections))


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Karyotype Viewer</title>
<style>
:root { --bg:#f6f6f3; --card:#fff; --ink:#1c1c1e; --muted:#6a6a70; --line:#e2e2de;
  --confirmed:#d62839; --own:#e08a00; --plain:#2a9d8f; --good:#2b7a4b; --bad:#c0392b; --warn:#b36b00; --neutral:#6a6a70; }
@media (prefers-color-scheme: dark) { :root { --bg:#131315; --card:#1d1d20; --ink:#ececef;
  --muted:#9a9aa3; --line:#2d2d32; --good:#4caf7a; --bad:#ef6a5b; --warn:#e0a040; --neutral:#9a9aa3; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width:1280px; margin:0 auto; padding:24px 20px 60px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:0 0 6px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.lede { color:var(--muted); margin:0 0 16px; max-width:760px; }
nav { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 14px; }
nav a { display:flex; align-items:center; gap:8px; padding:6px 12px; border:1px solid var(--line); border-radius:999px;
  background:var(--card); color:var(--ink); text-decoration:none; font-weight:600; }
nav a small { color:var(--muted); font-weight:400; }
.dot { width:9px; height:9px; border-radius:50%; background:var(--neutral); }
.dot.good,.badge.good { background:var(--good); } .dot.bad,.badge.bad { background:var(--bad); }
.dot.warn,.badge.warn { background:var(--warn); } .badge.neutral { background:var(--neutral); }
.badge { color:#fff; font-size:12px; font-weight:600; padding:2px 9px; border-radius:999px; }
.controls { display:flex; flex-wrap:wrap; gap:16px 24px; align-items:center; padding:12px 14px; margin:0 0 22px;
  background:var(--card); border:1px solid var(--line); border-radius:10px; }
.legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); }
.legend i { display:inline-block; width:18px; height:11px; margin-right:6px; vertical-align:-1px; border:2px solid; }
.legend .c { border-color:var(--confirmed); } .legend .o { border-color:var(--own); }
.legend .p { border-color:var(--plain); } .legend .e { border-color:var(--muted); border-style:dashed; }
label.toggle { display:flex; gap:6px; align-items:center; cursor:pointer; font-weight:600; }
section { margin:0 0 36px; scroll-margin-top:12px; }
.patient { padding:0 0 12px; border-bottom:1px solid var(--line); margin:0 0 14px; }
.patient p { margin:2px 0; } .sub { color:var(--muted); }
.k { display:inline-block; min-width:118px; color:var(--muted); }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(100%, 330px), 1fr)); gap:14px; }
.card { margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; cursor:zoom-in; }
.card.wide { grid-column:1 / -1; cursor:zoom-out; }
.card.wide svg { max-height:none; }
.card:focus-visible { outline:2px solid var(--plain); outline-offset:2px; }
svg { display:block; width:100%; height:auto; background:#000; }
figcaption { padding:10px 12px 12px; }
figcaption ul { list-style:none; margin:6px 0 0; padding:0; }
figcaption li { margin:3px 0; display:grid; grid-template-columns:122px 1fr; gap:8px; }
figcaption .k { min-width:0; } .miss { color:var(--bad); }
.cap-head { display:flex; justify-content:space-between; font-weight:700; } .count { color:var(--muted); font-weight:400; }
.box rect { fill:none; stroke-width:1.5px; vector-effect:non-scaling-stroke; }
.box.plain rect { stroke:var(--plain); stroke-opacity:.55; }
.box.own rect { stroke:var(--own); stroke-width:2.5px; }
.box.confirmed rect { stroke:var(--confirmed); stroke-width:3px; }
.box.extra rect { stroke-dasharray:6 4; }
.box text { font-weight:700; fill:#fff; stroke:#000; stroke-width:3px; paint-order:stroke; pointer-events:none; }
.box.plain text { display:none; }
body.all-labels .box.plain text { display:inline; font-weight:500; }
.box:hover rect { stroke-opacity:1; stroke-width:3px; }
.box:hover text { display:inline; }
</style></head>
<body><div class="wrap">
<h1>Karyotype Viewer</h1>
<p class="lede">Each image shows the pipeline's detected chromosomes and the chromosome the assignment solver gave each box.
Hover a box for details; click an image to enlarge it.</p>
<nav>{NAV}</nav>
<div class="controls">
  <div class="legend">
    <span><i class="c"></i>Confirmed across images</span>
    <span><i class="o"></i>Flagged by this image only</span>
    <span><i class="p"></i>Everything else</span>
    <span><i class="e"></i>Dashed: left over (extra copy)</span>
    <span>For a missing chromosome, the colored box is the copy that was found.</span>
  </div>
  <label class="toggle"><input type="checkbox" id="all"> Show every label</label>
</div>
{SECTIONS}
</div>
<script>
document.getElementById('all').addEventListener('change', e =>
  document.body.classList.toggle('all-labels', e.target.checked));
document.querySelectorAll('.card').forEach(card => {
  const toggle = () => card.classList.toggle('wide');
  card.addEventListener('click', toggle);
  card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
});
</script>
</body></html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="*", default=DEFAULT_CASES, help="patient case IDs (filename minus last digit)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="runs/viewer/karyotype_viewer.html")
    args = parser.parse_args()

    rows, details = analyze(args.cases, device=args.device)
    page = render(rows, details, args.cases)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {args.out} ({len(details)} images, {os.path.getsize(args.out) // 1024} KB)")
