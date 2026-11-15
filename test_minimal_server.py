import sys
import traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

class MinimalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f'Request: {self.path}', flush=True)
        if self.path == '/api/config':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            import json
            self.wfile.write(b'{"test": "ok"}')
        else:
            self.send_error(404)
    def log_message(self, fmt, *args):
        pass

print('Testing minimal handler...', flush=True)
server = ThreadingHTTPServer(('127.0.0.1', 8000), MinimalHandler)
server.daemon_threads = True

import threading
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()

import time
time.sleep(1)
print('Server thread started', flush=True)

import urllib.request
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=5)
    print('Response:', response.read().decode(), flush=True)
except Exception as e:
    print('Error:', e, flush=True)

server.shutdown()
server.server_close()
print('Done', flush=True)