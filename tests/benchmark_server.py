#!/usr/bin/env python3
"""
Simple HTTP server for the benchmark dashboard.

Serves static files and provides an API endpoint to start/stop benchmarks.

Usage:
    python benchmark_server.py [--port 8000]

Then open http://localhost:8000
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse

# Track the running benchmark process
_benchmark_process = None
_benchmark_lock = threading.Lock()


class BenchmarkHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files and handles benchmark API."""

    def do_POST(self):
        """Handle POST requests for benchmark API."""
        if self.path == '/api/benchmark/start':
            self.handle_start_benchmark()
        elif self.path == '/api/benchmark/stop':
            self.handle_stop_benchmark()
        else:
            self.send_error(404, 'Not Found')

    def do_GET(self):
        """Handle GET requests - serve static files or API."""
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '':
            # Serve dashboard at root
            self.path = '/benchmark_history.html'
            super().do_GET()
        elif parsed.path == '/api/benchmark/status':
            self.handle_benchmark_status()
        else:
            # Serve static files
            super().do_GET()

    def handle_start_benchmark(self):
        """Start a benchmark run in the background."""
        global _benchmark_process

        try:
            # Check if already running
            with _benchmark_lock:
                if _benchmark_process is not None and _benchmark_process.poll() is None:
                    self.send_response(409)  # Conflict
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'already_running',
                        'message': 'A benchmark is already running. Stop it first.'
                    }).encode())
                    return

            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            config = json.loads(body) if body else {}

            # Build command
            args = config.get('args', '--samples 100')
            # Use python3.10 explicitly - the engine requires it for dependencies
            python_exe = shutil.which('python3.10') or sys.executable
            cmd = f"{python_exe} benchmark_accuracy.py {args}"

            # Set up environment with PYTHONPATH to find hush_engine
            env = os.environ.copy()
            project_root = Path(__file__).parent.parent
            env['PYTHONPATH'] = str(project_root) + ':' + env.get('PYTHONPATH', '')

            # Start benchmark process
            # Write output to a log file instead of PIPE to avoid
            # deadlock when the pipe buffer fills (~64KB on macOS)
            log_path = Path(__file__).parent / 'benchmark_history' / 'benchmark_output.log'
            log_file = open(log_path, 'w')

            with _benchmark_lock:
                _benchmark_process = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=Path(__file__).parent,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group for clean termination
                )
                print(f"[API] Started benchmark (PID: {_benchmark_process.pid}): {cmd}")
                print(f"[API] Output log: {log_path}")

            # Monitor subprocess in background thread - update progress
            # if it crashes so the dashboard doesn't show stale "running"
            def _monitor_process(proc, log_fh):
                proc.wait()
                log_fh.close()
                exit_code = proc.returncode
                if exit_code != 0:
                    print(f"[API] Benchmark process exited with code {exit_code}")
                    progress_path = Path(__file__).parent / 'benchmark_history' / 'benchmark_progress.json'
                    try:
                        if progress_path.exists():
                            data = json.loads(progress_path.read_text())
                        else:
                            data = {}
                        if data.get('status') == 'running':
                            data['status'] = 'error'
                            data['phase'] = f'Process crashed (exit code {exit_code})'
                            progress_path.write_text(json.dumps(data, indent=2))
                            print(f"[API] Updated progress to error state")
                    except Exception as e:
                        print(f"[API] Failed to update progress on crash: {e}")
                else:
                    log_fh.close() if not log_fh.closed else None
                    print(f"[API] Benchmark completed successfully")

            monitor = threading.Thread(target=_monitor_process, args=(_benchmark_process, log_file), daemon=True)
            monitor.start()

            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'started',
                'message': 'Benchmark started in background',
                'command': cmd,
                'pid': _benchmark_process.pid
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'message': str(e)
            }).encode())

    def handle_stop_benchmark(self):
        """Stop the running benchmark."""
        global _benchmark_process

        try:
            with _benchmark_lock:
                pid = None

                # First, kill any running benchmark processes
                try:
                    subprocess.run(
                        "pkill -9 -f 'benchmark_accuracy.py'",
                        shell=True,
                        capture_output=True
                    )
                except Exception:
                    pass

                if _benchmark_process is not None and _benchmark_process.poll() is None:
                    pid = _benchmark_process.pid
                    try:
                        # Kill the process group
                        try:
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass

                        # Wait for process to terminate
                        try:
                            _benchmark_process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass

                    except ProcessLookupError:
                        pass  # Process already gone

                    _benchmark_process = None
                    print(f"[API] Stopped benchmark (PID: {pid})")

                # Wait a moment for process to fully stop writing
                time.sleep(1.0)

                # Now update progress file to stopped state (after process is dead)
                # Always write stopped state to ensure it takes effect
                progress_path = Path(__file__).parent / 'benchmark_history' / 'benchmark_progress.json'
                try:
                    if progress_path.exists():
                        data = json.loads(progress_path.read_text())
                    else:
                        data = {}
                    data['status'] = 'stopped'
                    data['phase'] = 'Stopped by user'
                    progress_path.write_text(json.dumps(data, indent=2))
                    print(f"[API] Updated progress file to stopped state")
                except Exception as e:
                    print(f"[API] Failed to update progress file: {e}")

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            msg = f'Benchmark stopped (PID: {pid})' if pid else 'Benchmark stopped'
            self.wfile.write(json.dumps({
                'status': 'stopped',
                'message': msg
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'message': str(e)
            }).encode())

    def handle_benchmark_status(self):
        """Return current benchmark status from progress file."""
        try:
            progress_path = Path(__file__).parent / 'benchmark_history' / 'benchmark_progress.json'
            if progress_path.exists():
                data = json.loads(progress_path.read_text())
            else:
                data = {'status': 'idle'}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'message': str(e)
            }).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def end_headers(self):
        """Add CORS headers to all responses."""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        """Custom log format."""
        first_arg = str(args[0]) if args else ''
        if '/api/' in first_arg:
            print(f"[API] {first_arg}")
        elif '.json' in first_arg:
            pass  # Suppress frequent JSON polling logs
        else:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Dashboard Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    args = parser.parse_args()

    # Change to tests directory
    import os
    os.chdir(Path(__file__).parent)

    server = HTTPServer(('', args.port), BenchmarkHandler)
    print(f"\n{'='*60}")
    print(f"  Hush Benchmark Dashboard")
    print(f"{'='*60}")
    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"\n  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
