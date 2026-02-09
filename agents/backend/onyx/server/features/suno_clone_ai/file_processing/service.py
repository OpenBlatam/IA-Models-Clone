"""File Processing Service - Servicio de procesamiento"""
from typing import Optional
from .base import BaseFileProcessor
from file_store.service import FileStoreService
from utils.service import UtilService

class FileProcessingService:
    def __init__(self, file_store: Optional[FileStoreService] = None, util_service: Optional[UtilService] = None):
        self.file_store = file_store
        self.util_service = util_service

