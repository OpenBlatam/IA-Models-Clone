"""
Advanced Document Processing System
==================================

Advanced document processing with support for multiple formats,
OCR, image analysis, and multimedia content extraction.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io
import base64
import hashlib
import mimetypes

# Document processing libraries
try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    from pdf2image import convert_from_path, convert_from_bytes
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False
    logging.warning("Document processing libraries not available")

# Text processing libraries
try:
    import textstat
    from readability import Document
    import textblob
    from textblob import TextBlob
    TEXT_PROCESSING_AVAILABLE = True
except ImportError:
    TEXT_PROCESSING_AVAILABLE = False
    logging.warning("Text processing libraries not available")

# Audio/Video processing
try:
    import librosa
    import soundfile as sf
    from moviepy.editor import VideoFileClip
    MEDIA_PROCESSING_AVAILABLE = True
except ImportError:
    MEDIA_PROCESSING_AVAILABLE = False
    logging.warning("Media processing libraries not available")

# Advanced NLP
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    logging.warning("Advanced NLP libraries not available")

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    filename: str
    file_size: int
    mime_type: str
    created_at: datetime
    modified_at: datetime
    language: Optional[str] = None
    encoding: Optional[str] = None
    pages: int = 0
    word_count: int = 0
    character_count: int = 0
    hash_md5: Optional[str] = None
    hash_sha256: Optional[str] = None

@dataclass
class TextAnalysis:
    """Text analysis results"""
    readability_score: float
    grade_level: str
    sentiment_polarity: float
    sentiment_subjectivity: float
    language: str
    word_frequency: Dict[str, int]
    key_phrases: List[str]
    named_entities: List[Dict[str, Any]]
    topics: List[str]
    summary: Optional[str] = None

@dataclass
class ImageAnalysis:
    """Image analysis results"""
    width: int
    height: int
    format: str
    mode: str
    has_text: bool
    text_content: Optional[str] = None
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    objects_detected: List[str] = field(default_factory=list)

@dataclass
class AudioAnalysis:
    """Audio analysis results"""
    duration: float
    sample_rate: int
    channels: int
    format: str
    has_speech: bool
    transcription: Optional[str] = None
    language: Optional[str] = None
    sentiment: Optional[float] = None

@dataclass
class ProcessedDocument:
    """Complete processed document"""
    metadata: DocumentMetadata
    text_content: str
    text_analysis: Optional[TextAnalysis] = None
    images: List[ImageAnalysis] = field(default_factory=list)
    audio: Optional[AudioAnalysis] = None
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

class AdvancedDocumentProcessor:
    """
    Advanced document processor with support for multiple formats
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize advanced document processor
        
        Args:
            models_dir: Directory for AI models
        """
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.sentence_model = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.summarizer = None
        
        self._load_models()
        
        # Supported formats
        self.supported_formats = {
            'text': ['.txt', '.md', '.rst', '.log'],
            'pdf': ['.pdf'],
            'documents': ['.docx', '.doc', '.odt', '.rtf'],
            'presentations': ['.pptx', '.ppt', '.odp'],
            'spreadsheets': ['.xlsx', '.xls', '.ods', '.csv'],
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
            'video': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz'],
            'web': ['.html', '.htm', '.xml', '.json']
        }
    
    def _load_models(self):
        """Load AI models for processing"""
        if not ADVANCED_NLP_AVAILABLE:
            return
        
        try:
            # Load sentence transformer for embeddings
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Load summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            logger.info("Advanced NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
    
    def detect_file_type(self, file_path: Union[str, Path]) -> str:
        """
        Detect file type based on extension and content
        
        Args:
            file_path: Path to file
            
        Returns:
            File type category
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        for file_type, extensions in self.supported_formats.items():
            if extension in extensions:
                return file_type
        
        return 'unknown'
    
    def calculate_file_hashes(self, file_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Calculate MD5 and SHA256 hashes of file
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (md5_hash, sha256_hash)
        """
        file_path = Path(file_path)
        
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return md5_hash.hexdigest(), sha256_hash.hexdigest()
    
    def extract_text_from_pdf(self, file_path: Union[str, Path]) -> Tuple[str, List[ImageAnalysis]]:
        """
        Extract text and images from PDF
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (text_content, image_analyses)
        """
        if not DOCUMENT_PROCESSING_AVAILABLE:
            return "", []
        
        text_content = ""
        image_analyses = []
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                page_text = page.get_text()
                text_content += page_text + "\n"
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            
                            # Analyze image
                            image_analysis = self.analyze_image_from_bytes(img_data)
                            if image_analysis:
                                image_analyses.append(image_analysis)
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
        
        return text_content, image_analyses
    
    def extract_text_from_image(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from image using OCR
        
        Args:
            file_path: Path to image file
            
        Returns:
            Extracted text
        """
        if not DOCUMENT_PROCESSING_AVAILABLE:
            return ""
        
        try:
            # Preprocess image for better OCR
            image = Image.open(file_path)
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply sharpening filter
            image = image.filter(ImageFilter.SHARPEN)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='eng+spa')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {e}")
            return ""
    
    def analyze_image_from_bytes(self, image_data: bytes) -> Optional[ImageAnalysis]:
        """
        Analyze image from bytes data
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Image analysis results
        """
        if not DOCUMENT_PROCESSING_AVAILABLE:
            return None
        
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))
            
            # Basic image properties
            width, height = image.size
            format_name = image.format or "unknown"
            mode = image.mode
            
            # Check if image has text using OCR
            has_text = False
            text_content = None
            
            try:
                text_content = pytesseract.image_to_string(image)
                has_text = len(text_content.strip()) > 0
            except:
                pass
            
            # Extract dominant colors
            dominant_colors = self._extract_dominant_colors(image)
            
            return ImageAnalysis(
                width=width,
                height=height,
                format=format_name,
                mode=mode,
                has_text=has_text,
                text_content=text_content,
                dominant_colors=dominant_colors
            )
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None
    
    def _extract_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image"""
        try:
            # Resize image for faster processing
            image = image.resize((150, 150))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get color data
            colors = image.getcolors(maxcolors=256*256*256)
            
            if not colors:
                return []
            
            # Sort by frequency and get top colors
            colors.sort(key=lambda x: x[0], reverse=True)
            dominant_colors = [color[1] for color in colors[:num_colors]]
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Error extracting dominant colors: {e}")
            return []
    
    def analyze_text(self, text: str) -> Optional[TextAnalysis]:
        """
        Perform comprehensive text analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Text analysis results
        """
        if not TEXT_PROCESSING_AVAILABLE or not text.strip():
            return None
        
        try:
            # Basic readability metrics
            readability_score = textstat.flesch_reading_ease(text)
            grade_level = textstat.text_standard(text)
            
            # Sentiment analysis
            blob = TextBlob(text)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity
            
            # Language detection
            language = blob.detect_language()
            
            # Word frequency
            words = blob.words
            word_frequency = {}
            for word in words:
                word_lower = word.lower()
                if len(word_lower) > 2:  # Filter short words
                    word_frequency[word_lower] = word_frequency.get(word_lower, 0) + 1
            
            # Sort by frequency
            word_frequency = dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:50])
            
            # Extract key phrases
            key_phrases = [phrase for phrase in blob.noun_phrases if len(phrase.split()) > 1][:20]
            
            # Named entity recognition
            named_entities = []
            if ADVANCED_NLP_AVAILABLE and self.ner_pipeline:
                try:
                    entities = self.ner_pipeline(text)
                    named_entities = [
                        {"text": entity["word"], "label": entity["entity_group"], "score": entity["score"]}
                        for entity in entities
                    ]
                except Exception as e:
                    logger.warning(f"Error in NER: {e}")
            
            # Topic extraction (simplified)
            topics = self._extract_topics(text)
            
            # Generate summary
            summary = None
            if ADVANCED_NLP_AVAILABLE and self.summarizer and len(text) > 100:
                try:
                    # Truncate text if too long
                    max_length = 1024
                    text_for_summary = text[:max_length] if len(text) > max_length else text
                    
                    summary_result = self.summarizer(text_for_summary, max_length=100, min_length=30, do_sample=False)
                    summary = summary_result[0]["summary_text"]
                except Exception as e:
                    logger.warning(f"Error generating summary: {e}")
            
            return TextAnalysis(
                readability_score=readability_score,
                grade_level=grade_level,
                sentiment_polarity=sentiment_polarity,
                sentiment_subjectivity=sentiment_subjectivity,
                language=language,
                word_frequency=word_frequency,
                key_phrases=key_phrases,
                named_entities=named_entities,
                topics=topics,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return None
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword extraction"""
        try:
            blob = TextBlob(text)
            
            # Get noun phrases as potential topics
            topics = []
            for phrase in blob.noun_phrases:
                if len(phrase.split()) <= 3 and len(phrase) > 5:
                    topics.append(phrase)
            
            # Remove duplicates and limit
            topics = list(set(topics))[:10]
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def analyze_audio(self, file_path: Union[str, Path]) -> Optional[AudioAnalysis]:
        """
        Analyze audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio analysis results
        """
        if not MEDIA_PROCESSING_AVAILABLE:
            return None
        
        try:
            # Load audio file
            y, sr = librosa.load(file_path)
            
            # Basic audio properties
            duration = len(y) / sr
            channels = 1  # librosa loads as mono
            format_name = Path(file_path).suffix.lower()
            
            # Detect speech (simple energy-based detection)
            has_speech = self._detect_speech(y, sr)
            
            # Transcription (placeholder - would need speech recognition)
            transcription = None
            
            # Language detection (placeholder)
            language = None
            
            # Sentiment from audio features (placeholder)
            sentiment = None
            
            return AudioAnalysis(
                duration=duration,
                sample_rate=sr,
                channels=channels,
                format=format_name,
                has_speech=has_speech,
                transcription=transcription,
                language=language,
                sentiment=sentiment
            )
            
        except Exception as e:
            logger.error(f"Error analyzing audio {file_path}: {e}")
            return None
    
    def _detect_speech(self, y: np.ndarray, sr: int) -> bool:
        """Detect if audio contains speech"""
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Calculate spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Simple heuristic: speech typically has moderate energy and spectral centroid
            avg_rms = np.mean(rms)
            avg_spectral_centroid = np.mean(spectral_centroids)
            
            # Thresholds (these would need tuning)
            speech_threshold_rms = 0.01
            speech_threshold_centroid = 1000
            
            return avg_rms > speech_threshold_rms and avg_spectral_centroid > speech_threshold_centroid
            
        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            return False
    
    async def process_document(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """
        Process document comprehensively
        
        Args:
            file_path: Path to document
            
        Returns:
            Processed document with all analysis
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        
        # Initialize result
        processed_doc = ProcessedDocument(
            metadata=DocumentMetadata(
                filename=file_path.name,
                file_size=file_path.stat().st_size,
                mime_type=mimetypes.guess_type(str(file_path))[0] or "unknown",
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                modified_at=datetime.fromtimestamp(file_path.stat().st_mtime)
            ),
            text_content=""
        )
        
        try:
            # Calculate file hashes
            md5_hash, sha256_hash = self.calculate_file_hashes(file_path)
            processed_doc.metadata.hash_md5 = md5_hash
            processed_doc.metadata.hash_sha256 = sha256_hash
            
            # Detect file type
            file_type = self.detect_file_type(file_path)
            
            # Process based on file type
            if file_type == 'pdf':
                text_content, images = self.extract_text_from_pdf(file_path)
                processed_doc.text_content = text_content
                processed_doc.images = images
                processed_doc.metadata.pages = len(images) if images else 0
                
            elif file_type == 'images':
                text_content = self.extract_text_from_image(file_path)
                processed_doc.text_content = text_content
                
                # Analyze the image itself
                image_analysis = self.analyze_image_from_bytes(file_path.read_bytes())
                if image_analysis:
                    processed_doc.images = [image_analysis]
                    
            elif file_type == 'audio':
                audio_analysis = self.analyze_audio(file_path)
                processed_doc.audio = audio_analysis
                
            elif file_type == 'text':
                # Simple text file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        processed_doc.text_content = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    with open(file_path, 'r', encoding='latin-1') as f:
                        processed_doc.text_content = f.read()
            
            # Analyze text content
            if processed_doc.text_content:
                processed_doc.text_analysis = self.analyze_text(processed_doc.text_content)
                processed_doc.metadata.word_count = len(processed_doc.text_content.split())
                processed_doc.metadata.character_count = len(processed_doc.text_content)
                
                if processed_doc.text_analysis:
                    processed_doc.metadata.language = processed_doc.text_analysis.language
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            processed_doc.processing_time = processing_time
            
        except Exception as e:
            error_msg = f"Error processing document {file_path}: {e}"
            logger.error(error_msg)
            processed_doc.errors.append(error_msg)
        
        return processed_doc
    
    def get_document_summary(self, processed_doc: ProcessedDocument) -> Dict[str, Any]:
        """
        Get comprehensive document summary
        
        Args:
            processed_doc: Processed document
            
        Returns:
            Document summary
        """
        summary = {
            "metadata": {
                "filename": processed_doc.metadata.filename,
                "file_size": processed_doc.metadata.file_size,
                "mime_type": processed_doc.metadata.mime_type,
                "language": processed_doc.metadata.language,
                "pages": processed_doc.metadata.pages,
                "word_count": processed_doc.metadata.word_count,
                "character_count": processed_doc.metadata.character_count,
                "hash_md5": processed_doc.metadata.hash_md5,
                "hash_sha256": processed_doc.metadata.hash_sha256
            },
            "content_analysis": {},
            "media_analysis": {},
            "processing_info": {
                "processing_time": processed_doc.processing_time,
                "errors": processed_doc.errors
            }
        }
        
        # Text analysis
        if processed_doc.text_analysis:
            summary["content_analysis"] = {
                "readability_score": processed_doc.text_analysis.readability_score,
                "grade_level": processed_doc.text_analysis.grade_level,
                "sentiment": {
                    "polarity": processed_doc.text_analysis.sentiment_polarity,
                    "subjectivity": processed_doc.text_analysis.sentiment_subjectivity
                },
                "language": processed_doc.text_analysis.language,
                "key_phrases": processed_doc.text_analysis.key_phrases[:10],
                "top_words": list(processed_doc.text_analysis.word_frequency.keys())[:10],
                "named_entities": processed_doc.text_analysis.named_entities[:10],
                "topics": processed_doc.text_analysis.topics,
                "summary": processed_doc.text_analysis.summary
            }
        
        # Image analysis
        if processed_doc.images:
            summary["media_analysis"]["images"] = [
                {
                    "dimensions": f"{img.width}x{img.height}",
                    "format": img.format,
                    "has_text": img.has_text,
                    "text_length": len(img.text_content) if img.text_content else 0,
                    "dominant_colors": len(img.dominant_colors)
                }
                for img in processed_doc.images
            ]
        
        # Audio analysis
        if processed_doc.audio:
            summary["media_analysis"]["audio"] = {
                "duration": processed_doc.audio.duration,
                "sample_rate": processed_doc.audio.sample_rate,
                "channels": processed_doc.audio.channels,
                "format": processed_doc.audio.format,
                "has_speech": processed_doc.audio.has_speech,
                "has_transcription": bool(processed_doc.audio.transcription)
            }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = AdvancedDocumentProcessor()
    
    # Test with a sample file
    test_file = "sample_document.pdf"
    
    if Path(test_file).exists():
        # Process document
        result = asyncio.run(processor.process_document(test_file))
        
        # Get summary
        summary = processor.get_document_summary(result)
        
        print("Document Processing Results:")
        print(f"Filename: {summary['metadata']['filename']}")
        print(f"File size: {summary['metadata']['file_size']} bytes")
        print(f"Word count: {summary['metadata']['word_count']}")
        print(f"Processing time: {summary['processing_info']['processing_time']:.2f}s")
        
        if summary['content_analysis']:
            print(f"Readability score: {summary['content_analysis']['readability_score']:.1f}")
            print(f"Sentiment polarity: {summary['content_analysis']['sentiment']['polarity']:.2f}")
            print(f"Key phrases: {summary['content_analysis']['key_phrases'][:5]}")
    
    print("Advanced document processor initialized successfully")



























