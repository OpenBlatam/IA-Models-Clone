#!/usr/bin/env python3
"""
File Naming Conventions for Video-OpusClip
Demonstrates proper naming using lowercase with underscores
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Example directory structure with proper naming
EXAMPLE_DIRECTORY_STRUCTURE = {
    "video_opusclip": {
        "core": {
            "video_processor.py",
            "audio_extractor.py",
            "frame_analyzer.py"
        },
        "security": {
            "authentication.py",
            "authorization.py",
            "input_validator.py",
            "rate_limiter.py",
            "encryption_manager.py"
        },
        "api": {
            "endpoints": {
                "video_endpoints.py",
                "user_endpoints.py",
                "admin_endpoints.py"
            },
            "middleware": {
                "cors_middleware.py",
                "auth_middleware.py",
                "logging_middleware.py"
            }
        },
        "database": {
            "models": {
                "user_model.py",
                "video_model.py",
                "session_model.py"
            },
            "migrations": {
                "001_initial_schema.py",
                "002_add_video_metadata.py"
            },
            "repositories": {
                "user_repository.py",
                "video_repository.py"
            }
        },
        "utils": {
            "file_utils.py",
            "string_utils.py",
            "date_utils.py",
            "validation_utils.py"
        },
        "tests": {
            "unit": {
                "test_video_processor.py",
                "test_authentication.py",
                "test_input_validator.py"
            },
            "integration": {
                "test_api_endpoints.py",
                "test_database_operations.py"
            },
            "fixtures": {
                "test_data.py",
                "mock_services.py"
            }
        },
        "config": {
            "settings.py",
            "database_config.py",
            "security_config.py"
        },
        "scripts": {
            "setup_database.py",
            "run_migrations.py",
            "backup_data.py"
        }
    }
}

# Example file naming patterns
FILE_NAMING_PATTERNS = {
    "python_modules": [
        "video_processor.py",
        "audio_extractor.py",
        "frame_analyzer.py",
        "user_authentication.py",
        "rate_limiter.py",
        "input_validator.py",
        "encryption_manager.py",
        "database_connection.py",
        "api_endpoints.py",
        "middleware_handler.py"
    ],
    "test_files": [
        "test_video_processor.py",
        "test_user_authentication.py",
        "test_input_validator.py",
        "test_api_endpoints.py",
        "test_database_operations.py"
    ],
    "configuration_files": [
        "database_config.py",
        "security_config.py",
        "api_config.py",
        "logging_config.py"
    ],
    "utility_files": [
        "file_utils.py",
        "string_utils.py",
        "date_utils.py",
        "validation_utils.py",
        "encryption_utils.py"
    ]
}

# Example class and function naming with underscores
class VideoProcessor:
    """Video processing with proper naming conventions"""
    
    def __init__(self, video_file_path: str):
        self.video_file_path = video_file_path
        self.processed_frames = []
        self.audio_track = None
    
    def extract_video_frames(self) -> List[str]:
        """Extract frames from video file"""
        # Implementation here
        return ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]
    
    def process_audio_track(self) -> str:
        """Process audio track from video"""
        # Implementation here
        return "processed_audio.wav"
    
    def generate_thumbnail(self) -> str:
        """Generate thumbnail from video"""
        # Implementation here
        return "video_thumbnail.jpg"

class UserAuthentication:
    """User authentication with proper naming conventions"""
    
    def __init__(self, database_connection):
        self.database_connection = database_connection
        self.current_user = None
        self.session_token = None
    
    def authenticate_user_credentials(self, username: str, password: str) -> bool:
        """Authenticate user with username and password"""
        # Implementation here
        return True
    
    def create_user_session(self, user_id: int) -> str:
        """Create new user session"""
        # Implementation here
        return "session_token_12345"
    
    def validate_session_token(self, token: str) -> bool:
        """Validate user session token"""
        # Implementation here
        return True

class InputValidator:
    """Input validation with proper naming conventions"""
    
    def __init__(self):
        self.validation_rules = {}
        self.error_messages = []
    
    def validate_email_address(self, email: str) -> bool:
        """Validate email address format"""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        import re
        errors = []
        
        if len(password) < 8:
            errors.append("Password too short")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password needs uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password needs lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password needs digit")
        
        return {
            "is_valid": len(errors) == 0,
            "error_messages": errors
        }
    
    def sanitize_user_input(self, input_text: str) -> str:
        """Sanitize user input"""
        import re
        # Remove dangerous characters
        sanitized = re.sub(r'[<>"\']', '', input_text)
        # Remove script tags
        sanitized = re.sub(r'<script.*?</script>', '', sanitized, flags=re.IGNORECASE)
        return sanitized.strip()

class RateLimiter:
    """Rate limiting with proper naming conventions"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests_per_window = max_requests
        self.time_window_seconds = time_window
        self.request_history = {}
    
    def is_request_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for client IP"""
        import time
        current_time = time.time()
        
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        
        # Remove old requests
        self.request_history[client_ip] = [
            req_time for req_time in self.request_history[client_ip]
            if current_time - req_time < self.time_window_seconds
        ]
        
        if len(self.request_history[client_ip]) >= self.max_requests_per_window:
            return False
        
        self.request_history[client_ip].append(current_time)
        return True
    
    def get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests for client IP"""
        import time
        current_time = time.time()
        
        if client_ip not in self.request_history:
            return self.max_requests_per_window
        
        valid_requests = [
            req_time for req_time in self.request_history[client_ip]
            if current_time - req_time < self.time_window_seconds
        ]
        
        return max(0, self.max_requests_per_window - len(valid_requests))

# Example utility functions with proper naming
def create_file_path(base_directory: str, file_name: str) -> str:
    """Create proper file path using underscores"""
    return os.path.join(base_directory, file_name.replace(" ", "_").lower())

def validate_file_name(file_name: str) -> bool:
    """Validate file name follows conventions"""
    import re
    # Check for proper naming pattern
    pattern = r'^[a-z][a-z0-9_]*\.py$'
    return bool(re.match(pattern, file_name))

def generate_test_file_name(module_name: str) -> str:
    """Generate test file name for module"""
    return f"test_{module_name}.py"

def create_directory_structure(base_path: str, structure: Dict[str, Any]) -> None:
    """Create directory structure with proper naming"""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):
            # It's a directory
            os.makedirs(path, exist_ok=True)
            create_directory_structure(path, content)
        else:
            # It's a file
            with open(path, 'w') as f:
                f.write(f"# {name}\n")

# Example configuration with proper naming
class DatabaseConfig:
    """Database configuration with proper naming"""
    
    def __init__(self):
        self.database_host = "localhost"
        self.database_port = 5432
        self.database_name = "video_opusclip"
        self.database_user = "postgres"
        self.database_password = "password"
        self.connection_timeout = 30
        self.max_connections = 100

class SecurityConfig:
    """Security configuration with proper naming"""
    
    def __init__(self):
        self.secret_key = "your-secret-key"
        self.encryption_key = "your-encryption-key"
        self.jwt_expiration_minutes = 30
        self.max_login_attempts = 5
        self.lockout_duration_seconds = 900
        self.rate_limit_requests = 100
        self.rate_limit_window_seconds = 60

class ApiConfig:
    """API configuration with proper naming"""
    
    def __init__(self):
        self.api_host = "0.0.0.0"
        self.api_port = 8000
        self.api_debug = False
        self.api_reload = True
        self.cors_origins = ["http://localhost:3000"]
        self.request_timeout_seconds = 30

# Example test structure with proper naming
class TestVideoProcessor:
    """Test class with proper naming conventions"""
    
    def test_extract_video_frames(self):
        """Test video frame extraction"""
        processor = VideoProcessor("test_video.mp4")
        frames = processor.extract_video_frames()
        assert len(frames) > 0
        assert all(frame.endswith('.jpg') for frame in frames)
    
    def test_process_audio_track(self):
        """Test audio track processing"""
        processor = VideoProcessor("test_video.mp4")
        audio_file = processor.process_audio_track()
        assert audio_file.endswith('.wav')
    
    def test_generate_thumbnail(self):
        """Test thumbnail generation"""
        processor = VideoProcessor("test_video.mp4")
        thumbnail = processor.generate_thumbnail()
        assert thumbnail.endswith('.jpg')

class TestUserAuthentication:
    """Test class for user authentication"""
    
    def test_authenticate_user_credentials(self):
        """Test user credential authentication"""
        auth = UserAuthentication(None)
        result = auth.authenticate_user_credentials("test_user", "test_password")
        assert isinstance(result, bool)
    
    def test_create_user_session(self):
        """Test user session creation"""
        auth = UserAuthentication(None)
        session_token = auth.create_user_session(123)
        assert isinstance(session_token, str)
        assert len(session_token) > 0

# Example main function with proper naming
def main():
    """Main function demonstrating proper naming conventions"""
    print("📁 File Naming Conventions Example")
    
    # Create example directory structure
    base_path = "example_project"
    create_directory_structure(base_path, EXAMPLE_DIRECTORY_STRUCTURE)
    
    # Demonstrate proper file naming
    print("✅ Proper file naming examples:")
    for category, files in FILE_NAMING_PATTERNS.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for file_name in files:
            is_valid = validate_file_name(file_name)
            status = "✅" if is_valid else "❌"
            print(f"   {status} {file_name}")
    
    # Demonstrate utility functions
    print("\n🔧 Utility functions:")
    file_path = create_file_path("utils", "file_utils.py")
    print(f"   Created path: {file_path}")
    
    test_file = generate_test_file_name("video_processor")
    print(f"   Test file: {test_file}")
    
    # Demonstrate configuration
    print("\n⚙️ Configuration examples:")
    db_config = DatabaseConfig()
    print(f"   Database: {db_config.database_name}")
    
    security_config = SecurityConfig()
    print(f"   JWT expiration: {security_config.jwt_expiration_minutes} minutes")
    
    api_config = ApiConfig()
    print(f"   API port: {api_config.api_port}")
    
    print("\n🎯 File naming conventions demonstrated!")

if __name__ == "__main__":
    main() 