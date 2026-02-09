"""
Rotating Logger for Piel Mejorador AI SAM3
==========================================

Advanced logging with rotation and compression.
"""

import logging
import logging.handlers
import gzip
import shutil
from pathlib import Path
from typing import Optional


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler with compression.
    
    Compresses old log files automatically.
    """
    
    def doRollover(self):
        """Override to add compression."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i + 1))
                if Path(sfn).exists():
                    if Path(dfn).exists():
                        Path(dfn).unlink()
                    Path(sfn).rename(dfn)
            
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if Path(dfn).exists():
                Path(dfn).unlink()
            
            # Compress current log
            if Path(self.baseFilename).exists():
                with open(self.baseFilename, 'rb') as f_in:
                    with gzip.open(dfn, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                Path(self.baseFilename).unlink()
        
        if not self.delay:
            self.stream = self._open()


def setup_rotating_logger(
    name: str,
    log_file: Path,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup rotating logger with compression.
    
    Args:
        name: Logger name
        log_file: Log file path
        max_bytes: Maximum bytes per log file
        backup_count: Number of backup files to keep
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Rotating file handler with compression
    file_handler = CompressedRotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    return logger




