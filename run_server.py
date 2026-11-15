import sys
import threading
import time
from splatlab.server import serve

print('Starting server in background...', flush=True)

def run_server():
    serve('127.0.0.1', 8000, False, 'auto')

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Give server time to start
time.sleep(2)

# Test the API
import urllib.request
try:
    response = urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=10)
    data = response.read().decode()
    print('API Response OK', flush=True)
    print(data[:200], flush=True)
except Exception as e:
    print('API Error:', e, flush=True)

# Keep main thread alive
try:
    server_thread.join()
except KeyboardInterrupt:
    print('Shutting down...', flush=True)