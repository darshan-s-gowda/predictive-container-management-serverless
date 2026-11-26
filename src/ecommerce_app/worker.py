"""
src/ecommerce_app/worker.py
=========================
Real HTTP serverless function worker subprocess.

Each container/worker runs as an isolated HTTP server on localhost:<port>,
handling e-commerce function invocations (/invoke) and health checks (/health).
Includes realistic runtime initialization delay on startup (cold start).
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PRODUCTS = [
    {"id": "P101", "name": "AeroBook Pro 14 Laptop", "category": "Laptops", "price": 1299.00, "stock": 18, "rating": 4.8, "img": "💻", "desc": "M-series chip, 16GB RAM, 512GB NVMe SSD, 18h battery"},
    {"id": "P102", "name": "Pulse ANC Wireless Headphones", "category": "Audio", "price": 249.00, "stock": 42, "rating": 4.7, "img": "🎧", "desc": "40dB adaptive noise cancellation, 35h playback, USB-C"},
    {"id": "P103", "name": "ChronoFit S4 Smartwatch", "category": "Wearables", "price": 329.00, "stock": 25, "rating": 4.6, "img": "⌚", "desc": "AMOLED LTPO display, ECG + SpO2, dual-band GPS"},
    {"id": "P104", "name": "MechKey 75% Tactile Keyboard", "category": "Accessories", "price": 139.00, "stock": 64, "rating": 4.9, "img": "⌨️", "desc": "Hot-swappable switches, PBT keycaps, gasket mount"},
    {"id": "P105", "name": "Lumina 4K Mirrorless Camera", "category": "Cameras", "price": 1549.00, "stock": 9, "rating": 4.8, "img": "📷", "desc": "26MP APS-C sensor, 4K60 10-bit video, 5-axis IBIS"},
    {"id": "P106", "name": "VoltGaN 100W 4-Port Charger", "category": "Accessories", "price": 69.00, "stock": 110, "rating": 4.7, "img": "🔌", "desc": "GaN III fast charging, 2x USB-C PD 3.0 + 2x USB-A"},
    {"id": "P107", "name": "SonicBar Mini Studio Speaker", "category": "Audio", "price": 179.00, "stock": 31, "rating": 4.5, "img": "🔊", "desc": "Spatial audio tuning, Bluetooth 5.3, optical + AUX"},
    {"id": "P108", "name": "UltraView 27\" 4K IPS Monitor", "category": "Displays", "price": 479.00, "stock": 14, "rating": 4.7, "img": "🖥️", "desc": "99% DCI-P3, 90W USB-C power delivery, height-adjustable"},
]

WORKER_ID = os.environ.get("WORKER_ID", "w-local")
BOOT_DELAY_MS = float(os.environ.get("COLD_START_INIT_MS", "900"))
BOOT_TIME = time.time()


def handle_function(fn_name: str, payload: dict) -> dict:
    """Execute the requested e-commerce serverless function."""
    if fn_name == "catalog":
        cat = str(payload.get("category", "")).strip().lower()
        items = [p for p in PRODUCTS if not cat or cat == "all" or p["category"].lower() == cat]
        return {"products": items, "count": len(items)}

    if fn_name == "search":
        q = str(payload.get("q", "")).strip().lower()
        items = [
            p for p in PRODUCTS
            if not q or q in p["name"].lower() or q in p["category"].lower() or q in p["desc"].lower()
        ]
        return {"products": items, "query": q, "count": len(items)}

    if fn_name == "recommend":
        pid = payload.get("product_id", "P101")
        recs = [p for p in PRODUCTS if p["id"] != pid][:3]
        return {"recommendations": recs}

    if fn_name == "checkout":
        # Simulate lightweight order validation + tax/total calculation
        items = payload.get("items", [])
        subtotal = sum(float(i.get("price", 0)) * int(i.get("qty", 1)) for i in items)
        tax = round(subtotal * 0.08, 2)
        total = round(subtotal + tax, 2)
        time.sleep(0.015)  # 15ms simulated payment/inventory check
        return {
            "order_id": f"ORD-{int(time.time() * 1000) % 1000000}",
            "status": "confirmed",
            "subtotal": round(subtotal, 2),
            "tax": tax,
            "total": total,
        }

    return {"ok": True, "fn": fn_name}


class WorkerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress noisy stdout logs

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "warm",
                "worker_id": WORKER_ID,
                "uptime_s": round(time.time() - BOOT_TIME, 2),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/invoke":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            req = json.loads(raw.decode("utf-8") or "{}")
            fn_name = req.get("fn", "catalog")
            payload = req.get("payload", {})
            t0 = time.perf_counter()
            result = handle_function(fn_name, payload)
            exec_ms = round((time.perf_counter() - t0) * 1000, 2)
            self._send_json(200, {
                "worker_id": WORKER_ID,
                "exec_ms": exec_ms,
                "result": result,
            })
        else:
            self._send_json(404, {"error": "not found"})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # Simulate container runtime + app initialization (cold start)
    if BOOT_DELAY_MS > 0:
        time.sleep(BOOT_DELAY_MS / 1000.0)
    server = ThreadingHTTPServer(("127.0.0.1", port), WorkerHandler)
    actual_port = server.server_address[1]
    # Signal readiness to parent orchestrator
    print(f"READY:{actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
