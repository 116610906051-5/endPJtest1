import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
METRICS_PATH = BASE / "performance_metrics.json"
OUT_PATH = BASE / "model_performance_dashboard.svg"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def to_points(fpr, tpr, left, top, width, height):
    pts = []
    for x, y in zip(fpr, tpr):
        px = left + (float(x) * width)
        py = top + ((1 - float(y)) * height)
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def main():
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    m1 = data["models"]["Stacking Ensemble"]
    m2 = data["models"]["BiLSTM"]

    W, H = 1700, 980

    # ROC canvas
    roc_x, roc_y, roc_w, roc_h = 860, 130, 780, 500

    roc1 = to_points(m1["roc_curve"]["fpr"], m1["roc_curve"]["tpr"], roc_x, roc_y, roc_w, roc_h)
    roc2 = to_points(m2["roc_curve"]["fpr"], m2["roc_curve"]["tpr"], roc_x, roc_y, roc_w, roc_h)

    def pct(v):
        return f"{v * 100:.2f}%"

    cm1 = m1["confusion_matrix"]
    cm2 = m2["confusion_matrix"]

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <style>
      .bg {{ fill: #f8fafc; }}
      .card {{ fill: #ffffff; stroke: #dbe3ee; stroke-width: 1.5; rx: 12; ry: 12; }}
      .title {{ font: 700 36px 'Segoe UI', Tahoma, sans-serif; fill: #0f172a; }}
      .h2 {{ font: 700 24px 'Segoe UI', Tahoma, sans-serif; fill: #0f172a; }}
      .h3 {{ font: 700 20px 'Segoe UI', Tahoma, sans-serif; fill: #0f172a; }}
      .txt {{ font: 500 16px 'Segoe UI', Tahoma, sans-serif; fill: #334155; }}
      .small {{ font: 500 14px 'Segoe UI', Tahoma, sans-serif; fill: #475569; }}
      .num {{ font: 700 30px 'Segoe UI', Tahoma, sans-serif; fill: #0f172a; }}
      .axis {{ stroke: #1e293b; stroke-width: 2; }}
      .grid {{ stroke: #dbe3ee; stroke-width: 1; }}
      .table-border {{ stroke: #cbd5e1; stroke-width: 1.2; fill: #fff; }}
    </style>
  </defs>

  <rect class="bg" x="0" y="0" width="{W}" height="{H}"/>

  <text class="title" x="60" y="65">Model Performance Dashboard</text>
  <text class="txt" x="60" y="98">Dataset rows: {data['dataset']['rows']} | Test rows: {data['dataset']['test_rows']} | Split: {esc(data['dataset']['split'])}</text>

  <!-- Left: KPI cards -->
  <rect class="card" x="40" y="130" width="780" height="260"/>
  <text class="h2" x="60" y="170">Key Metrics</text>

  <rect x="60" y="190" width="360" height="180" fill="#eff6ff" stroke="#bfdbfe" rx="12"/>
  <text class="h3" x="80" y="225" fill="#1d4ed8">Stacking Ensemble</text>
  <text class="small" x="80" y="255">Accuracy</text>
  <text class="num" x="80" y="290">{pct(m1['accuracy'])}</text>
  <text class="small" x="80" y="320">F1-weighted: {m1['f1_weighted']:.4f}</text>
  <text class="small" x="80" y="345">ROC-AUC: {m1['roc_auc']:.4f}</text>

  <rect x="440" y="190" width="360" height="180" fill="#ecfdf5" stroke="#a7f3d0" rx="12"/>
  <text class="h3" x="460" y="225" fill="#047857">BiLSTM</text>
  <text class="small" x="460" y="255">Accuracy</text>
  <text class="num" x="460" y="290">{pct(m2['accuracy'])}</text>
  <text class="small" x="460" y="320">F1-weighted: {m2['f1_weighted']:.4f}</text>
  <text class="small" x="460" y="345">ROC-AUC: {m2['roc_auc']:.4f}</text>

  <!-- Left middle: accuracy bars -->
  <rect class="card" x="40" y="410" width="780" height="220"/>
  <text class="h2" x="60" y="450">Accuracy Comparison</text>

  <text class="txt" x="60" y="495">Stacking Ensemble</text>
  <rect x="270" y="478" width="500" height="28" fill="#e2e8f0" rx="14"/>
  <rect x="270" y="478" width="{500 * float(m1['accuracy']):.2f}" height="28" fill="#2563eb" rx="14"/>
  <text class="txt" x="780" y="500" text-anchor="end">{pct(m1['accuracy'])}</text>

  <text class="txt" x="60" y="555">BiLSTM</text>
  <rect x="270" y="538" width="500" height="28" fill="#e2e8f0" rx="14"/>
  <rect x="270" y="538" width="{500 * float(m2['accuracy']):.2f}" height="28" fill="#059669" rx="14"/>
  <text class="txt" x="780" y="560" text-anchor="end">{pct(m2['accuracy'])}</text>

  <!-- Left bottom: confusion matrices -->
  <rect class="card" x="40" y="650" width="780" height="290"/>
  <text class="h2" x="60" y="690">Confusion Matrix</text>

  <text class="h3" x="60" y="730" fill="#1d4ed8">Stacking Ensemble</text>
  <rect class="table-border" x="60" y="745" width="340" height="170"/>
  <line x1="60" y1="790" x2="400" y2="790" stroke="#cbd5e1"/>
  <line x1="60" y1="850" x2="400" y2="850" stroke="#cbd5e1"/>
  <line x1="160" y1="745" x2="160" y2="915" stroke="#cbd5e1"/>
  <line x1="280" y1="745" x2="280" y2="915" stroke="#cbd5e1"/>
  <text class="small" x="220" y="775" text-anchor="middle">Pred 0</text>
  <text class="small" x="340" y="775" text-anchor="middle">Pred 1</text>
  <text class="small" x="110" y="825" text-anchor="middle">Actual 0</text>
  <text class="small" x="110" y="885" text-anchor="middle">Actual 1</text>
  <text class="txt" x="220" y="825" text-anchor="middle">{cm1[0][0]}</text>
  <text class="txt" x="340" y="825" text-anchor="middle">{cm1[0][1]}</text>
  <text class="txt" x="220" y="885" text-anchor="middle">{cm1[1][0]}</text>
  <text class="txt" x="340" y="885" text-anchor="middle">{cm1[1][1]}</text>

  <text class="h3" x="440" y="730" fill="#047857">BiLSTM</text>
  <rect class="table-border" x="440" y="745" width="340" height="170"/>
  <line x1="440" y1="790" x2="780" y2="790" stroke="#cbd5e1"/>
  <line x1="440" y1="850" x2="780" y2="850" stroke="#cbd5e1"/>
  <line x1="540" y1="745" x2="540" y2="915" stroke="#cbd5e1"/>
  <line x1="660" y1="745" x2="660" y2="915" stroke="#cbd5e1"/>
  <text class="small" x="600" y="775" text-anchor="middle">Pred 0</text>
  <text class="small" x="720" y="775" text-anchor="middle">Pred 1</text>
  <text class="small" x="490" y="825" text-anchor="middle">Actual 0</text>
  <text class="small" x="490" y="885" text-anchor="middle">Actual 1</text>
  <text class="txt" x="600" y="825" text-anchor="middle">{cm2[0][0]}</text>
  <text class="txt" x="720" y="825" text-anchor="middle">{cm2[0][1]}</text>
  <text class="txt" x="600" y="885" text-anchor="middle">{cm2[1][0]}</text>
  <text class="txt" x="720" y="885" text-anchor="middle">{cm2[1][1]}</text>

  <!-- Right: ROC chart -->
  <rect class="card" x="840" y="130" width="820" height="810"/>
  <text class="h2" x="860" y="170">ROC Curve</text>
  <text class="small" x="860" y="198">Blue: Stacking Ensemble (AUC {m1['roc_auc']:.4f}) | Green: BiLSTM (AUC {m2['roc_auc']:.4f})</text>

  <!-- ROC plotting area -->
  <rect x="{roc_x}" y="{roc_y}" width="{roc_w}" height="{roc_h}" fill="#fff" stroke="#cbd5e1"/>
'''

    # grid lines and labels
    for i in range(11):
        x = roc_x + (roc_w * i / 10)
        y = roc_y + (roc_h * i / 10)
        svg += f'  <line class="grid" x1="{x:.2f}" y1="{roc_y}" x2="{x:.2f}" y2="{roc_y + roc_h}"/>\n'
        svg += f'  <line class="grid" x1="{roc_x}" y1="{y:.2f}" x2="{roc_x + roc_w}" y2="{y:.2f}"/>\n'
        svg += f'  <text class="small" x="{x:.2f}" y="{roc_y + roc_h + 24}" text-anchor="middle">{i/10:.1f}</text>\n'
        svg += f'  <text class="small" x="{roc_x - 14}" y="{y + 5:.2f}" text-anchor="end">{1 - i/10:.1f}</text>\n'

    svg += f'''
  <line class="axis" x1="{roc_x}" y1="{roc_y + roc_h}" x2="{roc_x + roc_w}" y2="{roc_y + roc_h}"/>
  <line class="axis" x1="{roc_x}" y1="{roc_y}" x2="{roc_x}" y2="{roc_y + roc_h}"/>
  <line x1="{roc_x}" y1="{roc_y + roc_h}" x2="{roc_x + roc_w}" y2="{roc_y}" stroke="#64748b" stroke-width="2.2" stroke-dasharray="9 7"/>

  <polyline fill="none" stroke="#2563eb" stroke-width="3.2" points="{roc1}"/>
  <polyline fill="none" stroke="#059669" stroke-width="3.2" points="{roc2}"/>

  <text class="txt" x="{roc_x + roc_w/2}" y="{roc_y + roc_h + 55}" text-anchor="middle">False Positive Rate</text>
  <text class="txt" x="{roc_x - 55}" y="{roc_y + roc_h/2}" transform="rotate(-90 {roc_x - 55},{roc_y + roc_h/2})" text-anchor="middle">True Positive Rate</text>

  <!-- Legend -->
  <rect x="1220" y="665" width="420" height="120" fill="#fff" stroke="#cbd5e1" rx="10"/>
  <line x1="1245" y1="700" x2="1295" y2="700" stroke="#2563eb" stroke-width="4"/>
  <text class="txt" x="1310" y="706">Stacking Ensemble (AUC {m1['roc_auc']:.4f})</text>
  <line x1="1245" y1="735" x2="1295" y2="735" stroke="#059669" stroke-width="4"/>
  <text class="txt" x="1310" y="741">BiLSTM (AUC {m2['roc_auc']:.4f})</text>
  <line x1="1245" y1="770" x2="1295" y2="770" stroke="#64748b" stroke-width="3" stroke-dasharray="8 6"/>
  <text class="txt" x="1310" y="776">Baseline</text>
</svg>
'''

    OUT_PATH.write_text(svg, encoding="utf-8")
    print(f"Created: {OUT_PATH}")


if __name__ == "__main__":
    main()
