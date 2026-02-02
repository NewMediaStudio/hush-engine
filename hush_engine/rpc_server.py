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
import logging
import os
import stat
from pathlib import Path
from collections import deque
from time import time
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
from library_manager import get_library_manager
from locale_manager import get_locale_manager


# =============================================================================
# SECURITY: Audit Logging
# =============================================================================

def setup_audit_logger():
    """Configure audit logger for file operations."""
    hush_dir = Path.home() / ".hush"
    hush_dir.mkdir(parents=True, exist_ok=True)

    audit_log = logging.getLogger("hush.audit")
    audit_log.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not audit_log.handlers:
        log_file = hush_dir / "audit.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        audit_log.addHandler(handler)

        # Set restrictive permissions on audit log
        try:
            os.chmod(log_file, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except OSError:
            pass  # May fail if file doesn't exist yet

    return audit_log


# Initialize audit logger
audit_log = setup_audit_logger()


# =============================================================================
# SECURITY: Rate Limiting
# =============================================================================

class RateLimiter:
    """Simple rate limiter to prevent DoS attacks."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = deque()

    def check(self) -> bool:
        """Check if request is allowed. Returns False if rate limit exceeded."""
        now = time()
        # Remove old requests outside the window
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()
        if len(self.requests) >= self.max_requests:
            return False
        self.requests.append(now)
        return True


# =============================================================================
# SECURITY: Path Validation
# =============================================================================

# Allowed RPC methods (whitelist)
ALLOWED_METHODS = {
    'detectPII',
    'saveScrubbed',
    'getPDFPage',
    'getConfig',
    'saveConfig',
    'resetConfig',
    'ingestTrainingFeedback',
    # Library management
    'getLibraryStatus',
    'setLibraryEnabled',
    'downloadLibrary',
    # Locale management
    'getLocale',
    'setLocale',
}

# Protected locations that should never be written to
PROTECTED_PATHS = [
    '.ssh',
    '.gnupg',
    '.aws',
    '.credentials',
    'Library/Keychains',
    'Library/Cookies',
    '.password-store',
]


def validate_input_path(file_path: str) -> Path:
    """
    Validate an input file path for security.

    Raises ValueError if path is unsafe.

    NOTE: Audit logs only contain filenames (not full paths) to avoid
    exposing directory structure or potentially sensitive path components.
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    path = Path(file_path)

    # Resolve to absolute path (handles .. and symlinks)
    try:
        resolved = path.resolve(strict=True)  # strict=True requires file to exist
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except OSError as e:
        raise ValueError(f"Invalid path: {e}")

    # Check for symlink (after resolving, check original)
    if path.is_symlink():
        # SECURITY: Log only that symlink was blocked, not the path
        audit_log.warning("BLOCKED: symlink access attempt")
        raise ValueError("Symlinks are not allowed for security reasons")

    # Verify it's a regular file
    if not resolved.is_file():
        raise ValueError("Path must be a regular file")

    return resolved


def validate_output_path(file_path: str) -> Path:
    """
    Validate an output file path for security.

    Only allows writes to user's home directory (excluding protected locations)
    or system temp directories.

    NOTE: Audit logs only contain filenames (not full paths) to avoid
    exposing directory structure or potentially sensitive path components.
    """
    if not file_path:
        raise ValueError("Output path cannot be empty")

    path = Path(file_path)

    # Resolve parent directory (file may not exist yet)
    try:
        parent = path.parent.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"Output directory does not exist: {path.parent}")

    resolved = parent / path.name
    home = Path.home().resolve()

    # Allow writes to:
    # 1. User's home directory (but not protected locations)
    # 2. System temp directories
    allowed_prefixes = [
        str(home),
        '/tmp',
        '/var/folders',  # macOS temp
    ]

    resolved_str = str(resolved)
    if not any(resolved_str.startswith(prefix) for prefix in allowed_prefixes):
        # SECURITY: Log only that write was blocked, not the actual path
        audit_log.warning("BLOCKED: write attempt outside allowed locations")
        raise ValueError("Output must be in user's home directory or temp folder")

    # Check protected locations
    for protected in PROTECTED_PATHS:
        if protected in resolved_str:
            # SECURITY: Log only that protected location was blocked
            audit_log.warning(f"BLOCKED: write attempt to protected location ({protected})")
            raise ValueError(f"Cannot write to protected location containing: {protected}")

    return resolved


def validate_request(request: dict) -> None:
    """
    Validate JSON-RPC request structure.

    Raises ValueError if request is malformed.
    """
    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")

    # Validate ID
    req_id = request.get('id')
    if req_id is not None and not isinstance(req_id, (int, str)):
        raise ValueError("Request ID must be an integer or string")

    # Validate method
    method = request.get('method')
    if not isinstance(method, str):
        raise ValueError("Method must be a string")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Unknown method: {method}")

    # Validate params
    params = request.get('params')
    if params is not None and not isinstance(params, dict):
        raise ValueError("Params must be a dictionary")


class RPCServer:
    """JSON-RPC server for Hush backend operations"""

    def __init__(self):
        """Initialize the RPC server and warm up components"""
        sys.stderr.write("[RPCServer] Initializing...\n")
        sys.stderr.flush()

        self.router = FileRouter()
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

        # Warmup detector on init
        sys.stderr.write("[RPCServer] Warming up PII detector...\n")
        sys.stderr.flush()
        self.router.warmup()

        audit_log.info("RPC server initialized")
        sys.stderr.write("[RPCServer] Ready to accept requests\n")
        sys.stderr.flush()
    
    def handle_detect_pii(self, params):
        """Handle detectPII request"""
        file_path = params.get('filePath')

        # SECURITY: Validate input path
        validated_path = validate_input_path(file_path)
        file_path_str = str(validated_path)

        file_type = self.router.detect_file_type(file_path_str)

        # SECURITY: Log operation (path only, never contents/results)
        audit_log.info(f"DETECT | type={file_type} | file={validated_path.name}")

        if file_type == 'image':
            return self.router.detect_pii_image(file_path_str)
        elif file_type == 'spreadsheet':
            return self.router.detect_pii_spreadsheet(file_path_str)
        elif file_type == 'pdf':
            return self.router.detect_pii_pdf(file_path_str)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def handle_save_scrubbed(self, params):
        """Handle saveScrubbed request"""
        source = params.get('source')
        destination = params.get('destination')
        detections = params.get('detections')
        selected_indices = params.get('selectedIndices')
        optimize = params.get('optimize', False)  # Optional image optimization

        if not all([source, destination, detections is not None, selected_indices is not None]):
            raise ValueError("Missing required parameters")

        # SECURITY: Validate input and output paths
        validated_source = validate_input_path(source)
        validated_dest = validate_output_path(destination)

        source_str = str(validated_source)
        dest_str = str(validated_dest)

        file_type = self.router.detect_file_type(source_str)

        # SECURITY: Log operation (filenames only, never detection contents)
        num_redactions = len(selected_indices) if selected_indices else 0
        audit_log.info(f"SAVE | type={file_type} | src={validated_source.name} | dst={validated_dest.name} | redactions={num_redactions} | optimize={optimize}")

        if file_type == 'image':
            self.router.save_scrubbed_image(source_str, dest_str, detections, selected_indices, optimize=optimize)
        elif file_type == 'spreadsheet':
            self.router.save_scrubbed_spreadsheet(source_str, dest_str, detections, selected_indices)
        elif file_type == 'pdf':
            self.router.save_scrubbed_pdf(source_str, dest_str, detections, selected_indices, optimize=optimize)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return {"success": True}
    
    def handle_get_pdf_page(self, params):
        """Handle getPDFPage request"""
        file_path = params.get('filePath')
        page_num = params.get('pageNumber')
        optimize = params.get('optimize', False)  # Optional image optimization

        if page_num is None:
            raise ValueError("Missing required parameter: pageNumber")

        # SECURITY: Validate input path
        validated_path = validate_input_path(file_path)

        # SECURITY: Validate page number is reasonable
        if not isinstance(page_num, int) or page_num < 1 or page_num > 10000:
            raise ValueError("Page number must be an integer between 1 and 10000")

        return self.router.get_pdf_page_image(str(validated_path), page_num, optimize=optimize)
    
    def handle_get_config(self, params):
        """Handle getConfig request"""
        return detection_config.get_config().get_stats()
    
    def handle_save_config(self, params):
        """Handle saveConfig request"""
        thresholds = params.get('thresholds')
        enabled_entities = params.get('enabled_entities')

        if thresholds is None and enabled_entities is None:
            raise ValueError("At least one of thresholds or enabled_entities must be provided")

        config = detection_config.get_config()
        config.update_all(thresholds=thresholds, enabled_entities=enabled_entities)
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

    # =========================================================================
    # Library Management Methods
    # =========================================================================

    def handle_get_library_status(self, params):
        """Handle getLibraryStatus: get status of optional libraries"""
        library_name = params.get('libraryName')  # Optional, None returns all
        manager = get_library_manager()
        return manager.get_library_status(library_name)

    def handle_set_library_enabled(self, params):
        """Handle setLibraryEnabled: enable or disable a library"""
        library_name = params.get('libraryName')
        enabled = params.get('enabled')

        if library_name is None:
            raise ValueError("Missing required parameter: libraryName")
        if enabled is None:
            raise ValueError("Missing required parameter: enabled")
        if not isinstance(enabled, bool):
            raise ValueError("Parameter 'enabled' must be a boolean")

        manager = get_library_manager()
        result = manager.set_library_enabled(library_name, enabled)

        audit_log.info(f"LIBRARY | action={'enable' if enabled else 'disable'} | library={library_name}")

        return result

    def handle_download_library(self, params):
        """Handle downloadLibrary: install an optional library via pip"""
        library_name = params.get('libraryName')
        sync = params.get('sync', False)  # If True, wait for completion

        if library_name is None:
            raise ValueError("Missing required parameter: libraryName")

        manager = get_library_manager()

        audit_log.info(f"LIBRARY | action=download | library={library_name}")

        if sync:
            return manager.download_library_sync(library_name)
        else:
            return manager.download_library(library_name)

    # =========================================================================
    # Locale Management Methods
    # =========================================================================

    def handle_get_locale(self, params):
        """Handle getLocale: get current locale settings"""
        manager = get_locale_manager()
        return manager.get_locale()

    def handle_set_locale(self, params):
        """Handle setLocale: set locale preference"""
        locale = params.get('locale')

        if locale is None:
            raise ValueError("Missing required parameter: locale")
        if not isinstance(locale, str):
            raise ValueError("Parameter 'locale' must be a string")

        manager = get_locale_manager()
        result = manager.set_locale(locale)

        if result.get("success"):
            audit_log.info(f"LOCALE | action=set | locale={locale}")

        return result

    def handle_request(self, request):
        """Handle a single JSON-RPC request"""
        req_id = request.get('id')

        try:
            # SECURITY: Validate request structure
            validate_request(request)

            # SECURITY: Check rate limit
            if not self.rate_limiter.check():
                audit_log.warning(f"RATE_LIMIT exceeded for request {req_id}")
                return {
                    "id": req_id,
                    "error": {
                        "code": -429,
                        "message": "Rate limit exceeded. Please slow down."
                    }
                }

            method = request.get('method')
            params = request.get('params', {})

            sys.stderr.write(f"[RPCServer] Request {req_id}: {method}\n")
            sys.stderr.flush()

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
            # Library management
            elif method == 'getLibraryStatus':
                result = self.handle_get_library_status(params)
            elif method == 'setLibraryEnabled':
                result = self.handle_set_library_enabled(params)
            elif method == 'downloadLibrary':
                result = self.handle_download_library(params)
            # Locale management
            elif method == 'getLocale':
                result = self.handle_get_locale(params)
            elif method == 'setLocale':
                result = self.handle_set_locale(params)
            else:
                raise ValueError(f"Unknown method: {method}")

            sys.stderr.write(f"[RPCServer] Request {req_id} completed successfully\n")
            sys.stderr.flush()

            return {"id": req_id, "result": result}

        except Exception as e:
            # SECURITY: Log errors without exposing sensitive details
            error_type = type(e).__name__
            audit_log.warning(f"ERROR | id={req_id} | type={error_type}")

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
