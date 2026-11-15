import sys
import threading
import time
import socket
import urllib.request
from splatlab.server import serve

print('Starting serve in thread...', flush=True)

def run_server():
    serve('127.0.0.1', 8000, False, 'auto')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(2)
print('Server thread started', flush=True)

import urllib.request
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=10)
    print('Response:', response.read().decode()[:200], flush=True)
except Exception as e:
    print('Error:', e, flush=True)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect(('127.0.0.1', 8000))
    print('Socket connect: OK', flush=True)
    s.close()
except Exception as e:
    print('Socket connect failed:', e, flush=True)

print('Done', flush=True)