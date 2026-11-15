import sys
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from splatlab.server import MANAGER, resolve_device, device_report, label, tune_preset, PRESETS
from splatlab.mesh import backend_status
import json

class DebugHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f'Request: {self.path}', flush=True)
        if self.path == '/api/config':
            try:
                demo = None  # Simplified for testing
                device_error = None
                try:
                    dev = resolve_device('auto')
                    info = device_report(dev)
                    device_label = label(dev)
                    print(f'Device info: {info}', flush=True)
                except Exception as exc:
                    dev = None
                    device_error = str(exc)
                    info = {"type": "unavailable", "name": "Unavailable",
                            "vram_gb": None, "cuda": "", "error": device_error}
                    device_label = "GPU unavailable"
                presets = []
                for p in PRESETS.values():
                    tuned, _ = tune_preset(p, dev) if dev is not None else (p, [])
                    presets.append({
                        "key": tuned.key, "label": tuned.label,
                        "description": tuned.description,
                        "iterations": tuned.iterations,
                        "max_gaussians": tuned.max_gaussians,
                        "max_frames": tuned.max_frames,
                        "train_size": tuned.train_size,
                    })
                mesh_status = backend_status()
                print(f'Mesh status: {mesh_status}', flush=True)
                
                response = {
                    "version": "1.0.0",
                    "presets": presets,
                    "default_preset": "balanced",
                    "demo": demo,
                    "device": device_label,
                    "device_info": info,
                    "device_error": device_error,
                    "mesh_backend": mesh_status,
                }
                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                print(f'Error in handler: {e}', flush=True)
                traceback.print_exc()
                self.send_error(500, str(e))
        else:
            self.send_error(404)
    def log_message(self, fmt, *args):
        pass

print('Testing debug handler...', flush=True)
server = ThreadingHTTPServer(('127.0.0.1', 8000), DebugHandler)
server.daemon_threads = True

import threading
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()

import time
time.sleep(1)
print('Server thread started', flush=True)

import urllib.request
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=10)
    print('Response:', response.read().decode()[:200], flush=True)
except Exception as e:
    print('Error:', e, flush=True)

server.shutdown()
server.server_close()
print('Done', flush=True)