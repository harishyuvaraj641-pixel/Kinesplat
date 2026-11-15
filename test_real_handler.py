import sys
import traceback
from http.server import ThreadingHTTPServer
from splatlab.server import Handler, MANAGER

class TestHandler(Handler):
    def do_GET(self):
        print(f'RealHandler Request: {self.path}', flush=True)
        try:
            super().do_GET()
        except Exception as e:
            print(f'Handler error: {e}', flush=True)
            traceback.print_exc()
            self.send_error(500, str(e))

    def _api_get(self, path: str, query: dict) -> None:
        print(f'_api_get called: {path}', flush=True)
        try:
            super()._api_get(path, {})
        except Exception as e:
            print(f'_api_get error: {e}', flush=True)
            traceback.print_exc()
            self.send_error(500, str(e))

print('Testing real handler...', flush=True)
MANAGER.device = 'auto'
server = ThreadingHTTPServer(('127.0.0.1', 8000), TestHandler)
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