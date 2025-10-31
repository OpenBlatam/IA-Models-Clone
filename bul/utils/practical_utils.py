"""
BUL System - Practical Utilities
Real, practical utility functions for the BUL system
"""

import hashlib
import secrets
import string
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import uuid
import asyncio
from pathlib import Path
import mimetypes
import os

logger = logging.getLogger(__name__)

class SecurityUtils:
    """Real security utilities"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password"""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\']', '', text)
        text = text.strip()
        
        return text

class FileUtils:
    """Real file utilities"""
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension"""
        return Path(filename).suffix.lower()
    
    @staticmethod
    def get_mime_type(filename: str) -> str:
        """Get MIME type"""
        return mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    
    @staticmethod
    def is_allowed_file_type(filename: str, allowed_extensions: List[str]) -> bool:
        """Check if file type is allowed"""
        extension = FileUtils.get_file_extension(filename)
        return extension in allowed_extensions
    
    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """Generate unique filename"""
        extension = FileUtils.get_file_extension(original_filename)
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{extension}"
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0
    
    @staticmethod
    def create_directory_if_not_exists(directory_path: str) -> bool:
        """Create directory if it doesn't exist"""
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except OSError:
            return False

class TextUtils:
    """Real text utilities"""
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequency
        word_count = {}
        for word in filtered_words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        if not text:
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(TextUtils._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0.0, min(100.0, score))
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length"""
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def clean_html(html_text: str) -> str:
        """Remove HTML tags from text"""
        if not html_text:
            return ""
        
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html_text)

class ValidationUtils:
    """Real validation utilities"""
    
    @staticmethod
    def validate_required_fields(data: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validate required fields"""
        missing_fields = []
        
        for field in required_fields:
            if field not in data or not data[field]:
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    @staticmethod
    def validate_string_length(text: str, min_length: int = 0, max_length: int = None) -> Tuple[bool, str]:
        """Validate string length"""
        if not text:
            return False, "Text is required"
        
        if len(text) < min_length:
            return False, f"Text must be at least {min_length} characters long"
        
        if max_length and len(text) > max_length:
            return False, f"Text must be no more than {max_length} characters long"
        
        return True, ""
    
    @staticmethod
    def validate_numeric_range(value: Any, min_value: float = None, max_value: float = None) -> Tuple[bool, str]:
        """Validate numeric range"""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False, "Value must be a number"
        
        if min_value is not None and num_value < min_value:
            return False, f"Value must be at least {min_value}"
        
        if max_value is not None and num_value > max_value:
            return False, f"Value must be no more than {max_value}"
        
        return True, ""
    
    @staticmethod
    def validate_json_format(json_string: str) -> Tuple[bool, str]:
        """Validate JSON format"""
        try:
            json.loads(json_string)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}"

class DateUtils:
    """Real date utilities"""
    
    @staticmethod
    def format_datetime(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format datetime to string"""
        return dt.strftime(format_string)
    
    @staticmethod
    def parse_datetime(date_string: str, format_string: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
        """Parse string to datetime"""
        try:
            return datetime.strptime(date_string, format_string)
        except ValueError:
            return None
    
    @staticmethod
    def is_date_in_range(date: datetime, start_date: datetime, end_date: datetime) -> bool:
        """Check if date is in range"""
        return start_date <= date <= end_date
    
    @staticmethod
    def get_time_ago(dt: datetime) -> str:
        """Get human-readable time ago"""
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"

class CacheUtils:
    """Real cache utilities"""
    
    @staticmethod
    def generate_cache_key(prefix: str, *args) -> str:
        """Generate cache key"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    @staticmethod
    def serialize_data(data: Any) -> str:
        """Serialize data for caching"""
        return json.dumps(data, default=str)
    
    @staticmethod
    def deserialize_data(serialized_data: str) -> Any:
        """Deserialize cached data"""
        try:
            return json.loads(serialized_data)
        except json.JSONDecodeError:
            return None

class LoggingUtils:
    """Real logging utilities"""
    
    @staticmethod
    def setup_logging(level: str = "INFO", log_file: str = None):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
    
    @staticmethod
    def log_user_action(user_id: str, action: str, details: str = None):
        """Log user action"""
        logger.info(f"User {user_id} performed action: {action}" + (f" - {details}" if details else ""))
    
    @staticmethod
    def log_api_call(endpoint: str, method: str, user_id: str = None, status_code: int = None):
        """Log API call"""
        user_info = f" by user {user_id}" if user_id else ""
        status_info = f" - Status: {status_code}" if status_code else ""
        logger.info(f"API call: {method} {endpoint}{user_info}{status_info}")
    
    @staticmethod
    def log_error(error: Exception, context: str = None):
        """Log error with context"""
        context_info = f" in {context}" if context else ""
        logger.error(f"Error{context_info}: {str(error)}", exc_info=True)

class PerformanceUtils:
    """Real performance utilities"""
    
    @staticmethod
    async def measure_execution_time(func, *args, **kwargs):
        """Measure function execution time"""
        start_time = datetime.utcnow()
        result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        end_time = datetime.utcnow()
        
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        
        return result, execution_time
    
    @staticmethod
    def format_bytes(bytes_value: int) -> str:
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get memory usage information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss": memory_info.rss,  # Resident Set Size
                "vms": memory_info.vms,  # Virtual Memory Size
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}

class EmailUtils:
    """Real email utilities"""
    
    @staticmethod
    def validate_email_format(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def extract_domain(email: str) -> str:
        """Extract domain from email"""
        if '@' in email:
            return email.split('@')[1]
        return ""
    
    @staticmethod
    def is_business_email(email: str) -> bool:
        """Check if email is from business domain"""
        domain = EmailUtils.extract_domain(email)
        business_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        return domain not in business_domains













