import sys
import threading
import time
import socket
import urllib.request
import json
from splatlab.server import serve, MANAGER

print('Pre-loading in main thread...', flush=True)
MANAGER.device = 'auto'
_ = MANAGER.demo()
try:
    from splatlab.mesh import backend_status
    _ = backend_status()
except Exception:
    pass
print('Pre-loading done', flush=True)

print('Starting serve in thread...', flush=True)

def run_server():
    from splatlab.server import serve
    serve('127.0.0.1', 8000, False, 'auto')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(1)
print('Server thread started', flush=True)

import urllib.request
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=10)
    data = response.read().decode()
    j = json.loads(data)
    print('version:', j.get('version'), flush=True)
except Exception as e:
    print('Error:', e, flush=True)

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect(('127.0.0.1', 8000))
    print('Socket connect: OK', flush=True)
    s.close()
except Exception as e:
    print('Socket connect failed:', e, flush=True)

print('Done', flush=True)