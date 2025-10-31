"""
PDF Variantes Test Package
Paquete de pruebas para el sistema PDF Variantes
"""

from .test_system import TestPDFVariantesSystem
from .conftest import *

__all__ = [
    "TestPDFVariantesSystem",
    "test_settings",
    "temp_dir",
    "sample_pdf_content",
    "sample_text_content",
    "sample_document_data",
    "sample_variant_data",
    "sample_user_data"
]
