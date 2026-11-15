import sys
import threading
import time
import urllib.request
import json
from splatlab.server import serve, MANAGER

print('Starting serve in thread...', flush=True)

def run_server():
    serve('127.0.0.1', 8000, False, 'auto')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(2)
print('Server thread started', flush=True)

# Test multiple requests
for i in range(5):
    try:
        response = urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=5)
        data = response.read().decode()
        j = json.loads(data)
        print(f'API {i}: OK (version: {j.get("version")})', flush=True)
    except Exception as e:
        print(f'API {i}: Error: {e}', flush=True)
    time.sleep(0.5)

print('Done', flush=True)