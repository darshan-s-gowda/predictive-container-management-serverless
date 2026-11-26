"""
src/ecommerce_app/server.py
===========================
ShopPulse Serverless E-Commerce Website + Live Predictive Container Management Console.

Run:
    python src/ecommerce_app/server.py [port]
Default port: 8080 -> http://127.0.0.1:8080
Prometheus metrics: http://127.0.0.1:8080/metrics
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_app.orchestrator import LiveServerlessOrchestrator

ORCH: LiveServerlessOrchestrator | None = None

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ShopPulse — Serverless E-Commerce + Predictive Container Management</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --card2: #263349; --text: #f1f5f9;
    --muted: #94a3b8; --accent: #38bdf8; --green: #22c55e; --amber: #f59e0b;
    --red: #ef4444; --purple: #a855f7; --border: #334155;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); }
  header {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    padding: 14px 24px; background: #0b1120; border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 20;
  }
  .brand { font-size: 1.25rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
  .pill { font-size: 0.78rem; padding: 4px 10px; border-radius: 999px; font-weight: 600; }
  .pill-pred { background: rgba(34,197,94,0.18); color: var(--green); border: 1px solid rgba(34,197,94,0.4); }
  .pill-react { background: rgba(239,68,68,0.18); color: var(--red); border: 1px solid rgba(239,68,68,0.4); }
  .nav-tabs { display: flex; gap: 8px; }
  .tab-btn {
    background: var(--card); color: var(--text); border: 1px solid var(--border);
    padding: 8px 16px; border-radius: 8px; cursor: pointer; font-weight: 600;
  }
  .tab-btn.active { background: var(--accent); color: #0f172a; border-color: var(--accent); }
  .container { max-width: 1320px; margin: 0 auto; padding: 20px; }
  .hidden { display: none !important; }

  /* Storefront */
  .store-toolbar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 18px; }
  .search-input {
    flex: 1; min-width: 220px; padding: 10px 14px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--card); color: var(--text); font-size: 0.95rem;
  }
  .cat-btn {
    padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--card); color: var(--muted); cursor: pointer; font-size: 0.88rem;
  }
  .cat-btn.active { background: var(--accent); color: #0f172a; font-weight: 600; border-color: var(--accent); }
  .store-layout { display: grid; grid-template-columns: 1fr 340px; gap: 20px; align-items: start; }
  @media(max-width: 960px) { .store-layout { grid-template-columns: 1fr; } }
  .product-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 16px; }
  .product-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; display: flex; flex-direction: column; justify-content: space-between;
  }
  .prod-emoji { font-size: 2.4rem; margin-bottom: 8px; }
  .prod-title { font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
  .prod-desc { font-size: 0.82rem; color: var(--muted); margin-bottom: 12px; line-height: 1.35; }
  .prod-foot { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .price { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
  .btn {
    padding: 8px 12px; border-radius: 8px; border: none; cursor: pointer;
    font-weight: 600; font-size: 0.86rem; transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.9; }
  .btn-primary { background: var(--accent); color: #0f172a; }
  .btn-green { background: var(--green); color: #052e16; }
  .btn-amber { background: var(--amber); color: #451a03; }
  .btn-red { background: var(--red); color: #fff; }
  .btn-ghost { background: var(--card2); color: var(--text); border: 1px solid var(--border); }

  .cart-panel {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
    position: sticky; top: 82px;
  }
  .cart-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
  .receipt {
    margin-top: 12px; padding: 12px; border-radius: 8px; background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.35); font-size: 0.85rem; line-height: 1.45;
  }

  /* Live Console */
  .ctrl-bar {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 16px; margin-bottom: 18px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between;
  }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 18px; }
  .kpi {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }
  .kpi-label { font-size: 0.8rem; color: var(--muted); margin-bottom: 6px; }
  .kpi-val { font-size: 1.55rem; font-weight: 700; }
  .kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }

  .panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 18px; }
  .panel h3 { margin: 0 0 12px 0; font-size: 1rem; }
  .workers-wrap { display: flex; flex-wrap: wrap; gap: 10px; }
  .worker-chip {
    background: var(--card2); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; min-width: 155px; font-size: 0.82rem;
  }
  .worker-chip.warm { border-color: rgba(34,197,94,0.5); }
  .worker-chip.busy { border-color: rgba(245,158,11,0.7); }
  table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  .badge { padding: 2px 7px; border-radius: 5px; font-size: 0.75rem; font-weight: 600; }
  .b-ok { background: rgba(34,197,94,0.2); color: var(--green); }
  .b-warn { background: rgba(245,158,11,0.2); color: var(--amber); }
  .b-bad { background: rgba(239,68,68,0.2); color: var(--red); }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media(max-width: 960px) { .grid-2 { grid-template-columns: 1fr; } }
  a.link { color: var(--accent); text-decoration: none; font-size: 0.85rem; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>⚡ ShopPulse</span>
    <span style="font-size:0.82rem;color:var(--muted);font-weight:400;">Serverless E-Commerce Demo</span>
    <span id="modeBadge" class="pill pill-pred">Mode: Predictive</span>
  </div>
  <div class="nav-tabs">
    <button class="tab-btn active" id="tabStoreBtn" onclick="switchTab('store')">🛍️ E-Commerce Store</button>
    <button class="tab-btn" id="tabConsoleBtn" onclick="switchTab('console')">📊 Live Container Console</button>
    <a class="tab-btn" href="/metrics" target="_blank" style="text-decoration:none;">📈 /metrics</a>
  </div>
</header>

<div class="container">
  <!-- STOREFRONT TAB -->
  <div id="storeView">
    <div class="store-toolbar">
      <input id="searchBox" class="search-input" placeholder="Search laptops, headphones, cameras..." onkeyup="if(event.key==='Enter') loadProducts()"/>
      <button class="btn btn-primary" onclick="loadProducts()">Search</button>
      <div id="catFilters" style="display:flex;gap:6px;flex-wrap:wrap;">
        <button class="cat-btn active" onclick="setCategory('All', this)">All</button>
        <button class="cat-btn" onclick="setCategory('Laptops', this)">Laptops</button>
        <button class="cat-btn" onclick="setCategory('Audio', this)">Audio</button>
        <button class="cat-btn" onclick="setCategory('Wearables', this)">Wearables</button>
        <button class="cat-btn" onclick="setCategory('Accessories', this)">Accessories</button>
        <button class="cat-btn" onclick="setCategory('Cameras', this)">Cameras</button>
      </div>
    </div>

    <div class="store-layout">
      <div>
        <div id="apiBanner" style="margin-bottom:12px;font-size:0.83rem;color:var(--muted);">
          Every search, catalog load, and checkout hits a real serverless worker container.
        </div>
        <div id="productGrid" class="product-grid"></div>
      </div>

      <div class="cart-panel">
        <h3 style="margin-top:0;">🛒 Shopping Cart (<span id="cartCount">0</span>)</h3>
        <div id="cartItems"><span style="color:var(--muted);font-size:0.88rem;">Your cart is empty.</span></div>
        <div style="margin-top:14px;display:flex;justify-content:space-between;font-weight:700;">
          <span>Total</span>
          <span id="cartTotal">$0.00</span>
        </div>
        <button class="btn btn-green" style="width:100%;margin-top:12px;padding:10px;" onclick="checkout()">
          ⚡ Checkout (Invoke Serverless Function)
        </button>
        <div id="receiptBox" class="receipt hidden"></div>
      </div>
    </div>
  </div>

  <!-- LIVE CONSOLE TAB -->
  <div id="consoleView" class="hidden">
    <div class="ctrl-bar">
      <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
        <span style="font-weight:600;">Controller Mode:</span>
        <button id="btnPred" class="btn btn-green" onclick="setMode('predictive')">🧠 Predictive (TFT + Planner + PPO + Shield)</button>
        <button id="btnReact" class="btn btn-ghost" onclick="setMode('reactive')">🐢 Reactive (No Pre-warming)</button>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;">
        <button class="btn btn-amber" onclick="triggerBurst('flash_sale')">🚀 Trigger Flash Sale (24 concurrent reqs)</button>
        <button class="btn btn-primary" onclick="triggerBurst('browse')">🛒 Normal Browse Burst (8 reqs)</button>
      </div>
    </div>

    <div id="burstStatus" style="margin-bottom:14px;font-size:0.86rem;color:var(--accent);"></div>

    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Live Warm Containers</div>
        <div class="kpi-val" id="kWarm">-</div>
        <div class="kpi-sub" id="kPlanner">Planner: -</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Measured Mean / P95 Latency</div>
        <div class="kpi-val" id="kLat">-</div>
        <div class="kpi-sub">SLA target: &lt; 300 ms</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Real Cold Starts</div>
        <div class="kpi-val" id="kCold">-</div>
        <div class="kpi-sub" id="kSla">SLA violations: -</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Energy Proxy (Live Pool)</div>
        <div class="kpi-val" id="kEnergy">-</div>
        <div class="kpi-sub" id="kRps">Throughput: - req/s</div>
      </div>
    </div>

    <div class="panel">
      <h3>🐳 Live Worker Container Pool (Real OS HTTP Subprocesses)</h3>
      <div id="workersWrap" class="workers-wrap"></div>
    </div>

    <div class="grid-2">
      <div class="panel">
        <h3>🧠 Decision Pipeline Audit (Forecast → Planner → PPO → Safety Shield)</h3>
        <div style="overflow-x:auto;">
          <table>
            <thead><tr><th>Step</th><th>Mode</th><th>RPS / Fcst</th><th>Shield Action</th><th>Note</th></tr></thead>
            <tbody id="decisionBody"></tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <h3>📡 Recent Real HTTP Invocations</h3>
        <div style="overflow-x:auto;">
          <table>
            <thead><tr><th>Fn</th><th>Worker</th><th>Latency</th><th>Cold?</th><th>SLA</th></tr></thead>
            <tbody id="reqBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
let currentCategory = 'All';
let cart = [];

function switchTab(tab) {
  document.getElementById('storeView').classList.toggle('hidden', tab !== 'store');
  document.getElementById('consoleView').classList.toggle('hidden', tab !== 'console');
  document.getElementById('tabStoreBtn').classList.toggle('active', tab === 'store');
  document.getElementById('tabConsoleBtn').classList.toggle('active', tab === 'console');
}

function setCategory(cat, btn) {
  currentCategory = cat;
  document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  loadProducts();
}

async function loadProducts() {
  const q = document.getElementById('searchBox').value.trim();
  const url = q
    ? `/api/search?q=${encodeURIComponent(q)}`
    : `/api/catalog?category=${encodeURIComponent(currentCategory)}`;
  const res = await fetch(url);
  const data = await res.json();
  const products = data.data.products || [];
  document.getElementById('apiBanner').innerHTML =
    `Served by worker <code>${data.worker_id}</code> in <b>${data.latency_ms} ms</b> ` +
    (data.cold_start ? `<span class="badge b-warn">COLD START</span>` : `<span class="badge b-ok">WARM REUSE</span>`);

  const grid = document.getElementById('productGrid');
  grid.innerHTML = products.map(p => `
    <div class="product-card">
      <div>
        <div class="prod-emoji">${p.img}</div>
        <div class="prod-title">${p.name}</div>
        <div class="prod-desc">${p.desc}</div>
      </div>
      <div class="prod-foot">
        <span class="price">$${p.price.toFixed(2)}</span>
        <button class="btn btn-primary" onclick='addToCart(${JSON.stringify(p)})'>+ Add</button>
      </div>
    </div>
  `).join('');
}

function addToCart(p) {
  const existing = cart.find(i => i.id === p.id);
  if (existing) existing.qty += 1;
  else cart.push({ id: p.id, name: p.name, price: p.price, qty: 1 });
  renderCart();
}

function renderCart() {
  const count = cart.reduce((s, i) => s + i.qty, 0);
  const total = cart.reduce((s, i) => s + i.price * i.qty, 0);
  document.getElementById('cartCount').textContent = count;
  document.getElementById('cartTotal').textContent = `$${total.toFixed(2)}`;
  const box = document.getElementById('cartItems');
  if (!cart.length) {
    box.innerHTML = `<span style="color:var(--muted);font-size:0.88rem;">Your cart is empty.</span>`;
    return;
  }
  box.innerHTML = cart.map((i, idx) => `
    <div class="cart-item">
      <span>${i.name} × ${i.qty}</span>
      <span>$${(i.price * i.qty).toFixed(2)} <button class="btn btn-ghost" style="padding:2px 6px;margin-left:6px;" onclick="cart.splice(${idx},1);renderCart()">×</button></span>
    </div>
  `).join('');
}

async function checkout() {
  if (!cart.length) { alert('Add at least one item to cart first.'); return; }
  const res = await fetch('/api/checkout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items: cart })
  });
  const data = await res.json();
  const r = data.data;
  const box = document.getElementById('receiptBox');
  box.classList.remove('hidden');
  box.innerHTML = `
    <b>✅ Order Confirmed: ${r.order_id}</b><br>
    Total (incl. 8% tax): <b>$${r.total.toFixed(2)}</b><br>
    Worker: <code>${data.worker_id}</code> | Latency: <b>${data.latency_ms} ms</b> |
    ${data.cold_start ? '❄️ Cold Start' : '🔥 Warm Container'}
  `;
  cart = [];
  renderCart();
}

async function setMode(mode) {
  await fetch('/api/control/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  });
  refreshTelemetry();
}

async function triggerBurst(kind) {
  const el = document.getElementById('burstStatus');
  el.textContent = kind === 'flash_sale'
    ? '⏳ Running Flash Sale spike (forecasting + prewarming if Predictive, then 24 concurrent requests)...'
    : '⏳ Running 8 concurrent browse requests...';
  const res = await fetch('/api/control/burst', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind })
  });
  const summary = await res.json();
  el.innerHTML = `✅ Burst complete (${summary.mode}): <b>${summary.requests}</b> requests | ` +
    `Mean latency: <b>${summary.mean_latency_ms} ms</b> | Cold starts: <b>${summary.cold_starts}</b> | ` +
    `SLA violations (&gt;300ms): <b>${summary.sla_violations}</b>`;
  refreshTelemetry();
}

async function refreshTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    const t = await res.json();

    const badge = document.getElementById('modeBadge');
    badge.textContent = `Mode: ${t.mode === 'predictive' ? 'Predictive' : 'Reactive'}`;
    badge.className = `pill ${t.mode === 'predictive' ? 'pill-pred' : 'pill-react'}`;

    document.getElementById('btnPred').className = `btn ${t.mode === 'predictive' ? 'btn-green' : 'btn-ghost'}`;
    document.getElementById('btnReact').className = `btn ${t.mode === 'reactive' ? 'btn-red' : 'btn-ghost'}`;

    document.getElementById('kWarm').textContent = t.warm_containers;
    document.getElementById('kPlanner').textContent = `Planner: ${t.planner_backend}`;
    document.getElementById('kLat').textContent = `${t.mean_latency_ms} / ${t.p95_latency_ms} ms`;
    document.getElementById('kCold').textContent = `${t.total_cold_starts} (${t.cold_start_rate_pct}%)`;
    document.getElementById('kSla').textContent = `SLA violations: ${t.total_sla_violations} (${t.sla_violation_rate_pct}%)`;
    document.getElementById('kEnergy').textContent = t.energy_proxy;
    document.getElementById('kRps').textContent = `Throughput: ${t.current_rps} req/s | Total: ${t.total_requests}`;

    document.getElementById('workersWrap').innerHTML = t.workers.map(w => `
      <div class="worker-chip ${w.state}">
        <b>${w.worker_id}</b> <span class="badge ${w.state === 'warm' ? 'b-ok' : 'b-warn'}">${w.state}</span><br>
        Port: <code>:${w.port}</code> | Reqs: ${w.request_count}<br>
        Boot: ${w.boot_ms}ms | Idle: ${w.idle_s}s
      </div>
    `).join('');

    document.getElementById('decisionBody').innerHTML = t.decisions.map(d => `
      <tr>
        <td>#${d.step}</td>
        <td>${d.mode}</td>
        <td>${d.rps} / ${d.forecast_rps}</td>
        <td><span class="badge ${d.shield_overridden ? 'b-warn' : 'b-ok'}">${d.shield_approved}</span></td>
        <td>${d.exec_note}</td>
      </tr>
    `).join('');

    document.getElementById('reqBody').innerHTML = t.recent_requests.map(r => `
      <tr>
        <td><code>/api/${r.fn}</code></td>
        <td><code>${r.worker_id}</code></td>
        <td><b>${r.latency_ms} ms</b></td>
        <td>${r.cold_start ? '<span class="badge b-warn">COLD</span>' : '<span class="badge b-ok">WARM</span>'}</td>
        <td>${r.sla_violated ? '<span class="badge b-bad">VIOL</span>' : '<span class="badge b-ok">OK</span>'}</td>
      </tr>
    `).join('');
  } catch (e) {}
}

loadProducts();
refreshTelemetry();
setInterval(refreshTelemetry, 2000);
</script>
</body>
</html>
"""


class ShopPulseHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, code: int, text: str, content_type: str = "text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send_text(200, HTML_PAGE, "text/html; charset=utf-8")
            return

        if path == "/metrics":
            self._send_text(200, ORCH.prometheus_metrics(), "text/plain; version=0.0.4; charset=utf-8")
            return

        if path == "/api/telemetry":
            self._send_json(200, ORCH.get_telemetry())
            return

        if path == "/api/catalog":
            cat = qs.get("category", ["All"])[0]
            self._send_json(200, ORCH.invoke("catalog", {"category": cat}))
            return

        if path == "/api/search":
            q = qs.get("q", [""])[0]
            self._send_json(200, ORCH.invoke("search", {"q": q}))
            return

        if path == "/api/recommend":
            pid = qs.get("product_id", ["P101"])[0]
            self._send_json(200, ORCH.invoke("recommend", {"product_id": pid}))
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_json()

        if path == "/api/checkout":
            self._send_json(200, ORCH.invoke("checkout", body))
            return

        if path == "/api/control/mode":
            mode = body.get("mode", "predictive")
            ORCH.set_mode(mode)
            self._send_json(200, {"mode": mode})
            return

        if path == "/api/control/burst":
            kind = body.get("kind", "browse")
            n_reqs = 24 if kind == "flash_sale" else 8
            mode = ORCH.get_telemetry()["mode"]
            if kind == "flash_sale" and mode == "predictive":
                # Forecaster detects upcoming flash sale 1.5s before spike hits!
                ORCH.notify_upcoming_spike(duration_s=10.0, expected_rps=16.0)
                time.sleep(1.3)

            def one_call(i: int):
                fn = "checkout" if i % 4 == 0 else ("search" if i % 2 == 0 else "catalog")
                payload = {"items": [{"price": 199.0, "qty": 1}]} if fn == "checkout" else {"category": "All"}
                return ORCH.invoke(fn, payload)

            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(one_call, range(n_reqs)))

            lats = [r["latency_ms"] for r in results]
            colds = sum(1 for r in results if r["cold_start"])
            viols = sum(1 for r in results if r["sla_violated"])
            self._send_json(200, {
                "mode": mode,
                "requests": n_reqs,
                "mean_latency_ms": round(sum(lats) / len(lats), 1),
                "cold_starts": colds,
                "sla_violations": viols,
            })
            return

        self._send_json(404, {"error": "not found"})


def run_server(port: int = 8501):
    global ORCH
    host = os.environ.get("SHOPPULSE_HOST", "127.0.0.1")
    candidate_ports = [port, 8501, 8888, 9050, 5050, 0]
    seen = set()
    server = None
    bound_port = port
    for cand in candidate_ports:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            server = ThreadingHTTPServer((host, cand), ShopPulseHandler)
            bound_port = server.server_address[1]
            if cand != port:
                print(f"[ShopPulse] Port {port} unavailable on Windows; bound to port {bound_port} instead.", flush=True)
            break
        except OSError as e:
            if cand == 0:
                raise
            print(f"[ShopPulse] Could not bind {host}:{cand} ({e}); trying next port...", flush=True)

    ORCH = LiveServerlessOrchestrator(initial_workers=2)
    print(f"[ShopPulse] Live at http://127.0.0.1:{bound_port}", flush=True)
    print(f"[ShopPulse] Prometheus metrics at http://127.0.0.1:{bound_port}/metrics", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ORCH.shutdown()
        server.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(p)
