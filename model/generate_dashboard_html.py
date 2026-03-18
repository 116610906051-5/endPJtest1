import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
metrics_path = BASE_DIR / "performance_metrics.json"
output_path = BASE_DIR / "model_performance_dashboard.html"

if not metrics_path.exists():
    raise FileNotFoundError("performance_metrics.json not found. Run generate_performance_metrics.py first.")

with metrics_path.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

embedded = json.dumps(metrics, ensure_ascii=False)

html = f"""<!doctype html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Project Model Performance Dashboard</title>
  <style>
    :root {{
      --bg: #f1f5f9;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #e2e8f0;
      --blue: #2563eb;
      --green: #059669;
      --orange: #f59e0b;
      --red: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Segoe UI', Tahoma, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .container {{
      max-width: 1320px;
      margin: 24px auto;
      padding: 0 16px 40px;
    }}
    .header {{
      background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
      color: #fff;
      border-radius: 14px;
      padding: 20px 24px;
      margin-bottom: 16px;
      box-shadow: 0 8px 20px rgba(2, 6, 23, .15);
    }}
    .header h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .header p {{ margin: 0; opacity: .95; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, .05);
    }}
    .col-6 {{ grid-column: span 6; }}
    .col-12 {{ grid-column: span 12; }}
    .title {{ margin: 0 0 10px; font-size: 18px; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .kpi-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .kpi {{
      flex: 1;
      min-width: 130px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #f8fafc;
    }}
    .kpi .name {{ font-size: 12px; color: var(--muted); }}
    .kpi .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}

    .bar-wrap {{ margin: 10px 0 14px; }}
    .bar-label {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px; }}
    .bar {{ height: 14px; border-radius: 999px; background: #e2e8f0; overflow: hidden; }}
    .bar > span {{ display: block; height: 100%; }}

    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px; text-align: center; }}
    th {{ background: #f8fafc; }}

    .cm {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}

    .legend {{ display: flex; gap: 14px; font-size: 13px; margin-top: 8px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 5px; }}

    @media (max-width: 980px) {{
      .col-6 {{ grid-column: span 12; }}
      .cm {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="header">
      <h1>Model Performance Dashboard</h1>
      <p>Project: Fake News Detection | Dataset rows: <b id="rows"></b> | Test rows: <b id="testRows"></b></p>
      <p>Split: <b id="split"></b></p>
    </section>

    <section class="grid">
      <article class="card col-6" id="stacking-card"></article>
      <article class="card col-6" id="bilstm-card"></article>

      <article class="card col-6">
        <h3 class="title">Accuracy Comparison</h3>
        <div id="accuracy-bars"></div>
      </article>

      <article class="card col-6">
        <h3 class="title">ROC-AUC Comparison</h3>
        <div id="auc-bars"></div>
      </article>

      <article class="card col-12">
        <h3 class="title">ROC Curve</h3>
        <svg id="roc-svg" width="100%" viewBox="0 0 1050 460" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="legend">
          <span><i class="dot" style="background:#2563eb"></i>Stacking Ensemble</span>
          <span><i class="dot" style="background:#059669"></i>BiLSTM</span>
          <span><i class="dot" style="background:#64748b"></i>Baseline</span>
        </div>
      </article>

      <article class="card col-12">
        <h3 class="title">Metric Table</h3>
        <table id="metric-table"></table>
      </article>

      <article class="card col-12">
        <h3 class="title">Confusion Matrix</h3>
        <div class="cm" id="cm-wrap"></div>
      </article>
    </section>
  </div>

  <script>
    const data = {embedded};

    const fmtPct = (v) => (v * 100).toFixed(2) + '%';
    const round4 = (v) => Number(v).toFixed(4);

    const models = data.models;
    const names = Object.keys(models);

    document.getElementById('rows').textContent = data.dataset.rows;
    document.getElementById('testRows').textContent = data.dataset.test_rows;
    document.getElementById('split').textContent = data.dataset.split;

    function modelCard(name, color) {{
      const m = models[name];
      return `
        <h3 class="title">${{name}}</h3>
        <p class="sub">Saved model from this project</p>
        <div class="kpi-row">
          <div class="kpi"><div class="name">Accuracy</div><div class="val" style="color:${{color}}">${{fmtPct(m.accuracy)}}</div></div>
          <div class="kpi"><div class="name">F1 (weighted)</div><div class="val">${{round4(m.f1_weighted)}}</div></div>
          <div class="kpi"><div class="name">ROC-AUC</div><div class="val">${{round4(m.roc_auc)}}</div></div>
        </div>
      `;
    }}

    document.getElementById('stacking-card').innerHTML = modelCard('Stacking Ensemble', '#2563eb');
    document.getElementById('bilstm-card').innerHTML = modelCard('BiLSTM', '#059669');

    function barSection(elId, key, colors) {{
      const wrap = document.getElementById(elId);
      wrap.innerHTML = names.map((n, i) => {{
        const v = models[n][key];
        return `
          <div class="bar-wrap">
            <div class="bar-label"><span>${{n}}</span><b>${{key === 'accuracy' ? fmtPct(v) : round4(v)}}</b></div>
            <div class="bar"><span style="width:${{Math.max(1, v*100)}}%; background:${{colors[i]}}"></span></div>
          </div>
        `;
      }}).join('');
    }}

    barSection('accuracy-bars', 'accuracy', ['#2563eb', '#059669']);
    barSection('auc-bars', 'roc_auc', ['#2563eb', '#059669']);

    // Metric table
    const table = document.getElementById('metric-table');
    table.innerHTML = `
      <thead>
        <tr>
          <th>Model</th><th>Accuracy</th><th>Precision (weighted)</th><th>Recall (weighted)</th><th>F1 (weighted)</th><th>ROC-AUC</th>
        </tr>
      </thead>
      <tbody>
        ${{names.map(n => `
          <tr>
            <td><b>${{n}}</b></td>
            <td>${{fmtPct(models[n].accuracy)}}</td>
            <td>${{round4(models[n].precision_weighted)}}</td>
            <td>${{round4(models[n].recall_weighted)}}</td>
            <td>${{round4(models[n].f1_weighted)}}</td>
            <td>${{round4(models[n].roc_auc)}}</td>
          </tr>
        `).join('')}}
      </tbody>
    `;

    // Confusion matrix
    const cmWrap = document.getElementById('cm-wrap');
    cmWrap.innerHTML = names.map(n => {{
      const cm = models[n].confusion_matrix;
      return `
        <div>
          <h4 style="margin:0 0 8px">${{n}}</h4>
          <table>
            <thead><tr><th></th><th>Pred: Fake (0)</th><th>Pred: Real (1)</th></tr></thead>
            <tbody>
              <tr><th>Actual: Fake (0)</th><td>${{cm[0][0]}}</td><td>${{cm[0][1]}}</td></tr>
              <tr><th>Actual: Real (1)</th><td>${{cm[1][0]}}</td><td>${{cm[1][1]}}</td></tr>
            </tbody>
          </table>
        </div>
      `;
    }}).join('');

    // ROC SVG
    const svg = document.getElementById('roc-svg');
    const W = 1050, H = 460;
    const m = {{left: 70, right: 22, top: 20, bottom: 54}};
    const PW = W - m.left - m.right;
    const PH = H - m.top - m.bottom;

    const sx = x => m.left + x * PW;
    const sy = y => m.top + (1 - y) * PH;

    function line(points, color, width=3, dash='') {{
      return `<polyline fill="none" stroke="${{color}}" stroke-width="${{width}}" stroke-dasharray="${{dash}}" points="${{points}}" />`;
    }}

    let parts = [];

    // Grid and axes
    for (let i=0; i<=10; i++) {{
      const x = m.left + i * (PW/10);
      const y = m.top + i * (PH/10);
      parts.push(`<line x1="${{x}}" y1="${{m.top}}" x2="${{x}}" y2="${{m.top+PH}}" stroke="#e2e8f0"/>`);
      parts.push(`<line x1="${{m.left}}" y1="${{y}}" x2="${{m.left+PW}}" y2="${{y}}" stroke="#e2e8f0"/>`);

      parts.push(`<text x="${{x}}" y="${{m.top+PH+24}}" font-size="12" fill="#334155" text-anchor="middle">${{(i/10).toFixed(1)}}</text>`);
      parts.push(`<text x="${{m.left-10}}" y="${{y+4}}" font-size="12" fill="#334155" text-anchor="end">${{(1-i/10).toFixed(1)}}</text>`);
    }}

    parts.push(`<line x1="${{m.left}}" y1="${{m.top+PH}}" x2="${{m.left+PW}}" y2="${{m.top+PH}}" stroke="#0f172a" stroke-width="2"/>`);
    parts.push(`<line x1="${{m.left}}" y1="${{m.top}}" x2="${{m.left}}" y2="${{m.top+PH}}" stroke="#0f172a" stroke-width="2"/>`);

    // Baseline diagonal
    parts.push(line(`${{sx(0)}},${{sy(0)}} ${{sx(1)}},${{sy(1)}}`, '#64748b', 2.4, '8 6'));

    function downsample(arrX, arrY, maxN=140) {{
      if (arrX.length <= maxN) return [arrX, arrY];
      const step = arrX.length / maxN;
      const nx = [], ny = [];
      for (let i=0; i<maxN; i++) {{
        const idx = Math.min(arrX.length - 1, Math.floor(i * step));
        nx.push(arrX[idx]);
        ny.push(arrY[idx]);
      }}
      return [nx, ny];
    }}

    const colorMap = {{'Stacking Ensemble': '#2563eb', 'BiLSTM': '#059669'}};
    names.forEach(n => {{
      const c = models[n].roc_curve;
      const [xf, yt] = downsample(c.fpr, c.tpr, 180);
      const pts = xf.map((x, i) => `${{sx(x)}},${{sy(yt[i])}}`).join(' ');
      parts.push(line(pts, colorMap[n] || '#f59e0b', 3));
    }});

    parts.push(`<text x="${{W/2}}" y="${{H-10}}" text-anchor="middle" font-size="14" fill="#0f172a">False Positive Rate</text>`);
    parts.push(`<text x="18" y="${{H/2}}" transform="rotate(-90 18,${{H/2}})" text-anchor="middle" font-size="14" fill="#0f172a">True Positive Rate</text>`);

    svg.innerHTML = parts.join('');
  </script>
</body>
</html>
"""

output_path.write_text(html, encoding="utf-8")
print(f"Created: {output_path}")
