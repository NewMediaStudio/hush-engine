#!/usr/bin/env python3
"""
RPC Server for Hush - Runs as subprocess, communicates via JSON-RPC over stdin/stdout

Protocol:
- Request: {"id": <int>, "method": <string>, "params": <dict>}
- Response: {"id": <int>, "result": <any>}
- Error: {"id": <int>, "error": {"code": <int>, "message": <string>}}
"""

import sys
import json
import traceback
from pathlib import Path
import builtins

# Save the original stdout for JSON-RPC communication
_rpc_stdout = sys.stdout
# Redirect sys.stdout to stderr so that any unintended print calls from 
# libraries (like Presidio or pandas) go to the diagnostic log instead of breaking RPC
sys.stdout = sys.stderr

def print_to_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    _original_print(*args, **kwargs)

# Replace builtins.print as well just to be thorough
_original_print = builtins.print
builtins.print = print_to_stderr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ui.file_router import FileRouter
import detection_config
from analyze_feedback import FeedbackAnalyzer


class RPCServer:
    """JSON-RPC server for Hush backend operations"""
    
    def __init__(self):
        """Initialize the RPC server and warm up components"""
        sys.stderr.write("[RPCServer] Initializing...\n")
        sys.stderr.flush()
        
        self.router = FileRouter()
        
        # Warmup detector on init
        sys.stderr.write("[RPCServer] Warming up PII detector...\n")
        sys.stderr.flush()
        self.router.warmup()
        
        sys.stderr.write("[RPCServer] Ready to accept requests\n")
        sys.stderr.flush()
    
    def handle_detect_pii(self, params):
        """Handle detectPII request"""
        file_path = params.get('filePath')
        if not file_path:
            raise ValueError("Missing required parameter: filePath")
        
        file_type = self.router.detect_file_type(file_path)
        
        if file_type == 'image':
            return self.router.detect_pii_image(file_path)
        elif file_type == 'spreadsheet':
            return self.router.detect_pii_spreadsheet(file_path)
        elif file_type == 'pdf':
            return self.router.detect_pii_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def handle_save_scrubbed(self, params):
        """Handle saveScrubbed request"""
        source = params.get('source')
        destination = params.get('destination')
        detections = params.get('detections')
        selected_indices = params.get('selectedIndices')
        
        if not all([source, destination, detections is not None, selected_indices is not None]):
            raise ValueError("Missing required parameters")
        
        file_type = self.router.detect_file_type(source)
        
        if file_type == 'image':
            self.router.save_scrubbed_image(source, destination, detections, selected_indices)
        elif file_type == 'spreadsheet':
            self.router.save_scrubbed_spreadsheet(source, destination, detections, selected_indices)
        elif file_type == 'pdf':
            self.router.save_scrubbed_pdf(source, destination, detections, selected_indices)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return {"success": True}
    
    def handle_get_pdf_page(self, params):
        """Handle getPDFPage request"""
        file_path = params.get('filePath')
        page_num = params.get('pageNumber')
        
        if not file_path or page_num is None:
            raise ValueError("Missing required parameters: filePath and pageNumber")
        
        return self.router.get_pdf_page_image(file_path, page_num)
    
    def handle_get_config(self, params):
        """Handle getConfig request"""
        return detection_config.get_config().get_stats()
    
    def handle_save_config(self, params):
        """Handle saveConfig request"""
        thresholds = params.get('thresholds')
        if thresholds is None:
            raise ValueError("Missing required parameter: thresholds")
        
        detection_config.save_config(thresholds)
        return {"success": True}
    
    def handle_reset_config(self, params):
        """Handle resetConfig request: restore shipped defaults and clear training data"""
        detection_config.reset_config()
        return {"success": True}

    def handle_ingest_training_feedback(self, params):
        """Handle ingestTrainingFeedback: read ~/.hush/training_feedback.jsonl and adjust thresholds"""
        feedback_path = Path.home() / ".hush" / "training_feedback.jsonl"
        if not feedback_path.exists():
            return {"status": "no_data", "adjustments": [], "message": "No feedback file yet"}
        analyzer = FeedbackAnalyzer(str(feedback_path))
        result = analyzer.auto_adjust_thresholds(min_samples=3)
        return result
    
    def handle_request(self, request):
        """Handle a single JSON-RPC request"""
        req_id = request.get('id')
        method = request.get('method')
        params = request.get('params', {})
        
        sys.stderr.write(f"[RPCServer] Request {req_id}: {method}\n")
        sys.stderr.flush()
        
        try:
            # Dispatch to method handler
            if method == 'detectPII':
                result = self.handle_detect_pii(params)
            elif method == 'saveScrubbed':
                result = self.handle_save_scrubbed(params)
            elif method == 'getConfig':
                result = self.handle_get_config(params)
            elif method == 'saveConfig':
                result = self.handle_save_config(params)
            elif method == 'resetConfig':
                result = self.handle_reset_config(params)
            elif method == 'ingestTrainingFeedback':
                result = self.handle_ingest_training_feedback(params)
            elif method == 'getPDFPage':
                result = self.handle_get_pdf_page(params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            sys.stderr.write(f"[RPCServer] Request {req_id} completed successfully\n")
            sys.stderr.flush()
            
            return {"id": req_id, "result": result}
            
        except Exception as e:
            sys.stderr.write(f"[RPCServer] Request {req_id} failed: {e}\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
            
            return {
                "id": req_id,
                "error": {
                    "code": -1,
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
            }
    
    def run(self):
        """Main loop: read requests from stdin, write responses to stdout"""
        sys.stderr.write("[RPCServer] Entering main loop\n")
        sys.stderr.flush()
        
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    
                    # Write response as single line JSON to the dedicated RPC stdout
                    response_json = json.dumps(response)
                    _rpc_stdout.write(response_json + "\n")
                    _rpc_stdout.flush()
                    
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"[RPCServer] Invalid JSON: {e}\n")
                    sys.stderr.flush()
                    # Write response as single line JSON to the dedicated RPC stdout
                    response_json = json.dumps(error_response)
                    _rpc_stdout.write(response_json + "\n")
                    _rpc_stdout.flush()
                    
        except KeyboardInterrupt:
            sys.stderr.write("[RPCServer] Interrupted by user\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[RPCServer] Fatal error: {e}\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
            sys.exit(1)


def main():
    """Entry point"""
    server = RPCServer()
    server.run()


if __name__ == '__main__':
    main()
