"""
Test Result Encryption
Encrypt sensitive test results for security
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os


class TestResultEncryption:
    """Encrypt and decrypt test results"""
    
    def __init__(self, project_root: Path, password: str = None):
        self.project_root = project_root
        self.key_file = project_root / ".test_encryption_key"
        self.key = self._get_or_create_key(password)
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self, password: str = None) -> bytes:
        """Get or create encryption key"""
        if self.key_file.exists():
            # Load existing key
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Create new key
            if password:
                # Derive key from password
                salt = os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            else:
                # Generate random key
                key = Fernet.generate_key()
            
            # Save key
            self.key_file.parent.mkdir(exist_ok=True)
            with open(self.key_file, 'wb') as f:
                f.write(key)
            
            return key
    
    def encrypt_result(
        self,
        result_data: Dict,
        output_file: Path = None
    ) -> Path:
        """Encrypt test result data"""
        # Convert to JSON string
        json_data = json.dumps(result_data, ensure_ascii=False)
        
        # Encrypt
        encrypted_data = self.cipher.encrypt(json_data.encode('utf-8'))
        
        # Save to file
        if output_file is None:
            timestamp = result_data.get('timestamp', datetime.now().isoformat())[:10]
            output_file = self.project_root / "test_results_encrypted" / f"result_{timestamp}.enc"
            output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted_data)
        
        print(f"✅ Encrypted result saved to {output_file}")
        return output_file
    
    def decrypt_result(
        self,
        encrypted_file: Path
    ) -> Dict:
        """Decrypt test result file"""
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt
        decrypted_data = self.cipher.decrypt(encrypted_data)
        json_data = decrypted_data.decode('utf-8')
        
        # Parse JSON
        result_data = json.loads(json_data)
        
        return result_data
    
    def encrypt_directory(
        self,
        source_dir: Path,
        target_dir: Path = None
    ) -> int:
        """Encrypt all results in a directory"""
        if target_dir is None:
            target_dir = source_dir.parent / f"{source_dir.name}_encrypted"
        
        target_dir.mkdir(exist_ok=True)
        
        encrypted_count = 0
        
        for result_file in source_dir.glob("*.json"):
            try:
                # Load result
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # Encrypt
                encrypted_file = target_dir / f"{result_file.stem}.enc"
                self.encrypt_result(result_data, encrypted_file)
                encrypted_count += 1
            except Exception as e:
                print(f"Error encrypting {result_file}: {e}")
        
        print(f"✅ Encrypted {encrypted_count} files")
        return encrypted_count


def main():
    """CLI interface"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Test Result Encryption')
    parser.add_argument('--encrypt', type=str, help='Encrypt result file')
    parser.add_argument('--decrypt', type=str, help='Decrypt result file')
    parser.add_argument('--encrypt-dir', type=str, help='Encrypt directory')
    parser.add_argument('--password', type=str, help='Encryption password')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    encryption = TestResultEncryption(project_root, args.password)
    
    if args.encrypt:
        print(f"🔒 Encrypting: {args.encrypt}")
        with open(args.encrypt, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        encryption.encrypt_result(result_data)
    elif args.decrypt:
        print(f"🔓 Decrypting: {args.decrypt}")
        result = encryption.decrypt_result(Path(args.decrypt))
        print(f"✅ Decrypted: {result.get('run_name', 'unknown')}")
    elif args.encrypt_dir:
        print(f"🔒 Encrypting directory: {args.encrypt_dir}")
        encryption.encrypt_directory(Path(args.encrypt_dir))
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

