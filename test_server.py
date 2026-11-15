import sys
import threading
import time
from splatlab.server import serve

print('Starting server...', flush=True)

def run_server():
    serve('127.0.0.1', 8000, False, 'auto')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(2)
print('Testing...', flush=True)

import urllib.request
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=10)
    data = response.read().decode()
    print('API Response OK', flush=True)
    import json
    j = json.loads(data)
    print('version:', j.get('version'), flush=True)
except Exception as e:
    print('API Error:', e, flush=True)

print('Done', flush=True)