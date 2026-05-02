import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

print(f"🦞 Serving installers at http://localhost:{PORT}")
print(f"   Windows: iwr -useb http://localhost:{PORT}/install.ps1 | iex")
print(f"   Mac/Linux: curl -fsSL http://localhost:{PORT}/install.sh | bash")
print("\nPress Ctrl+C to stop.")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
