"""Enhanced PDF processor using best-in-class libraries."""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from functools import partial
import asyncio
import time
import logging
from pathlib import Path
import hashlib
import json

# Best PDF libraries
import fitz  # PyMuPDF - fastest PDF library
import pdfplumber  # Best for text extraction
import PyPDF2  # Backup PDF library
from pdf2image import convert_from_bytes  # PDF to image
from PIL import Image  # Image processing

# AI/ML libraries
import spacy
import nltk
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import textstat  # Readability metrics
from rake_nltk import Rake  # Keyword extraction

# Performance libraries
import httpx  # Async HTTP
import aiofiles  # Async file operations
from tenacity import retry, stop_after_attempt, wait_exponential  # Retry logic

# Data processing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import gensim
from gensim import corpora, models

from .advanced_performance import intelligent_cache, performance_monitor
from .advanced_error_handling import intelligent_error_handler, ErrorSeverity, ErrorCategory

logger = logging.getLogger(__name__)

# Initialize NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found, using basic processing")
    nlp = None

# Initialize sentence transformer
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    logger.warning("Sentence transformer not available")
    sentence_model = None

# Initialize keyword extractor
rake = Rake()


@intelligent_cache(maxsize=1000, ttl=600.0)
@performance_monitor("enhanced_content_extraction")
@intelligent_error_handler(ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING, "content_extraction")
async def enhanced_extract_content(file_content: bytes) -> Dict[str, Any]:
    """Enhanced PDF content extraction using best libraries."""
    
    # Use PyMuPDF for fast processing
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        
        # Extract text with PyMuPDF
        text_content = ""
        for page in doc:
            text_content += page.get_text()
        
        # Extract metadata
        metadata = doc.metadata
        page_count = doc.page_count
        
        doc.close()
        
    except Exception as e:
        logger.warning(f"PyMuPDF failed, trying pdfplumber: {e}")
        
        # Fallback to pdfplumber
        try:
            import io
            pdf = pdfplumber.open(io.BytesIO(file_content))
            
            text_content = ""
            for page in pdf.pages:
                text_content += page.extract_text() or ""
            
            metadata = pdf.metadata or {}
            page_count = len(pdf.pages)
            pdf.close()
            
        except Exception as e2:
            logger.error(f"All PDF extraction methods failed: {e2}")
            raise ValueError(f"PDF extraction failed: {e2}")
    
    # Extract images if needed
    images = []
    try:
        images_data = convert_from_bytes(file_content, dpi=150)
        images = [img for img in images_data[:3]]  # Limit to first 3 pages
    except Exception as e:
        logger.warning(f"Image extraction failed: {e}")
    
    return {
        "text_content": text_content,
        "metadata": metadata,
        "page_count": page_count,
        "file_hash": hashlib.md5(file_content).hexdigest(),
        "file_size": len(file_content),
        "images": images,
        "extraction_method": "enhanced_multi_library"
    }


@intelligent_cache(maxsize=500, ttl=1800.0)
@performance_monitor("enhanced_topic_extraction")
async def enhanced_extract_topics(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced topic extraction using advanced NLP libraries."""
    text = content_data.get("text_content", "")
    if not text:
        return {"topics": [], "main_topic": None, "confidence": 0.0}
    
    topics = []
    
    # Method 1: RAKE keyword extraction
    try:
        rake.extract_keywords_from_text(text)
        rake_keywords = rake.get_ranked_phrases()[:10]
        
        for keyword in rake_keywords:
            topics.append({
                "topic": keyword,
                "method": "rake",
                "confidence": 0.8
            })
    except Exception as e:
        logger.warning(f"RAKE extraction failed: {e}")
    
    # Method 2: spaCy named entity recognition
    if nlp:
        try:
            doc = nlp(text[:10000])  # Limit text length
            entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
            
            for entity, label in entities[:10]:
                topics.append({
                    "topic": entity,
                    "method": "spacy_ner",
                    "type": label,
                    "confidence": 0.7
                })
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}")
    
    # Method 3: TF-IDF based topic extraction
    try:
        sentences = text.split('.')[:50]  # Limit for performance
        if len(sentences) > 5:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-10:][::-1]
            
            for idx in top_indices:
                topics.append({
                    "topic": feature_names[idx],
                    "method": "tfidf",
                    "score": float(scores[idx]),
                    "confidence": 0.6
                })
    except Exception as e:
        logger.warning(f"TF-IDF extraction failed: {e}")
    
    # Method 4: LDA topic modeling
    try:
        if len(text.split()) > 100:  # Need sufficient text
            sentences = text.split('.')[:100]
            texts = [sentence.split() for sentence in sentences if len(sentence.split()) > 5]
            
            if len(texts) > 10:
                dictionary = corpora.Dictionary(texts)
                corpus = [dictionary.doc2bow(text) for text in texts]
                
                lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=2)
                
                for topic_id, topic_words in lda_model.print_topics(num_words=3):
                    words = topic_words.split('"')[1::2]  # Extract words
                    for word in words[:3]:
                        topics.append({
                            "topic": word.strip(),
                            "method": "lda",
                            "topic_id": topic_id,
                            "confidence": 0.5
                        })
    except Exception as e:
        logger.warning(f"LDA topic modeling failed: {e}")
    
    # Remove duplicates and rank by confidence
    unique_topics = {}
    for topic in topics:
        topic_text = topic["topic"].lower()
        if topic_text not in unique_topics or topic["confidence"] > unique_topics[topic_text]["confidence"]:
            unique_topics[topic_text] = topic
    
    final_topics = list(unique_topics.values())
    final_topics.sort(key=lambda x: x["confidence"], reverse=True)
    
    return {
        "topics": final_topics[:15],
        "main_topic": final_topics[0]["topic"] if final_topics else None,
        "total_topics": len(final_topics),
        "confidence": sum(t["confidence"] for t in final_topics) / len(final_topics) if final_topics else 0,
        "extraction_methods": list(set(t["method"] for t in final_topics))
    }


@intelligent_cache(maxsize=200, ttl=3600.0)
@performance_monitor("enhanced_variant_generation")
async def enhanced_generate_variants(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced variant generation using AI and advanced text processing."""
    text = content_data.get("text_content", "")
    if not text:
        return {"variants": [], "generation_method": "enhanced_ai"}
    
    variants = {}
    
    # Smart summary using text statistics
    try:
        sentences = text.split('.')
        if len(sentences) > 3:
            # Use textstat for readability analysis
            sentence_scores = []
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    try:
                        readability = textstat.flesch_reading_ease(sentence)
                        sentence_scores.append((sentence.strip(), readability))
                    except:
                        sentence_scores.append((sentence.strip(), 50))  # Default score
            
            # Select most readable sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            summary_sentences = [s[0] for s in sentence_scores[:3]]
            variants["summary"] = '. '.join(summary_sentences) + '.'
        else:
            variants["summary"] = text[:200] + "..." if len(text) > 200 else text
    except Exception as e:
        logger.warning(f"Smart summary generation failed: {e}")
        variants["summary"] = text[:200] + "..." if len(text) > 200 else text
    
    # Intelligent outline generation
    try:
        if nlp and len(text) > 500:
            doc = nlp(text[:5000])  # Limit for performance
            sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 20]
            
            # Extract key sentences based on entities and importance
            key_sentences = []
            for sent in sentences[:20]:
                entities = [ent.text for ent in sent.ents]
                if entities:  # Sentences with entities are more important
                    key_sentences.append(sent.text.strip())
            
            if not key_sentences:
                key_sentences = sentences[:5]
            
            variants["outline"] = "\n".join([f"- {s}" for s in key_sentences[:10]])
        else:
            sentences = text.split('.')
            variants["outline"] = "\n".join([f"- {s.strip()}" for s in sentences[:5] if s.strip()])
    except Exception as e:
        logger.warning(f"Intelligent outline generation failed: {e}")
        sentences = text.split('.')
        variants["outline"] = "\n".join([f"- {s.strip()}" for s in sentences[:5] if s.strip()])
    
    # AI-powered highlights
    try:
        if sentence_model and len(text) > 200:
            sentences = text.split('.')
            if len(sentences) > 5:
                # Use sentence embeddings to find most representative sentences
                sentence_embeddings = sentence_model.encode(sentences[:20])
                
                # Find centroid
                centroid = np.mean(sentence_embeddings, axis=0)
                
                # Find sentences closest to centroid
                distances = np.linalg.norm(sentence_embeddings - centroid, axis=1)
                closest_indices = np.argsort(distances)[:3]
                
                highlights = [sentences[i].strip() for i in closest_indices if sentences[i].strip()]
                variants["highlights"] = " | ".join(highlights)
            else:
                variants["highlights"] = text[:100] + "..." if len(text) > 100 else text
        else:
            variants["highlights"] = text[:100] + "..." if len(text) > 100 else text
    except Exception as e:
        logger.warning(f"AI-powered highlights failed: {e}")
        variants["highlights"] = text[:100] + "..." if len(text) > 100 else text
    
    # Generate word cloud data
    try:
        if len(text.split()) > 50:
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            variants["word_cloud"] = [{"word": word, "frequency": freq} for word, freq in top_words]
    except Exception as e:
        logger.warning(f"Word cloud generation failed: {e}")
    
    return {
        "variants": variants,
        "variant_count": len(variants),
        "generation_method": "enhanced_ai_nlp",
        "ai_features": {
            "spacy_used": nlp is not None,
            "sentence_transformer_used": sentence_model is not None,
            "textstat_used": True
        }
    }


@intelligent_cache(maxsize=200, ttl=3600.0)
@performance_monitor("enhanced_quality_analysis")
async def enhanced_analyze_quality(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced quality analysis using advanced text statistics."""
    text = content_data.get("text_content", "")
    
    if not text:
        return {"quality_score": 0.0, "metrics": {}, "analysis_method": "enhanced_textstat"}
    
    metrics = {}
    
    # Basic text statistics
    metrics["word_count"] = len(text.split())
    metrics["sentence_count"] = len(text.split('.'))
    metrics["paragraph_count"] = len(text.split('\n\n'))
    
    # Advanced readability metrics using textstat
    try:
        metrics["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        metrics["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        metrics["gunning_fog"] = textstat.gunning_fog(text)
        metrics["smog_index"] = textstat.smog_index(text)
        metrics["automated_readability_index"] = textstat.automated_readability_index(text)
        metrics["coleman_liau_index"] = textstat.coleman_liau_index(text)
        metrics["linsear_write_formula"] = textstat.linsear_write_formula(text)
        metrics["dale_chall_readability_score"] = textstat.dale_chall_readability_score(text)
        
        # Calculate average readability
        readability_scores = [
            metrics["flesch_reading_ease"],
            metrics["flesch_kincaid_grade"],
            metrics["gunning_fog"],
            metrics["smog_index"],
            metrics["automated_readability_index"],
            metrics["coleman_liau_index"],
            metrics["linsear_write_formula"],
            metrics["dale_chall_readability_score"]
        ]
        
        metrics["average_readability"] = np.mean(readability_scores)
        
    except Exception as e:
        logger.warning(f"Textstat analysis failed: {e}")
        metrics["average_readability"] = 50  # Default score
    
    # Text complexity analysis
    try:
        words = text.split()
        sentences = text.split('.')
        
        metrics["avg_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
        metrics["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0
        
        # Syllable analysis
        metrics["avg_syllables_per_word"] = textstat.avg_syllables_per_word(text)
        
        # Complex word analysis
        complex_words = [word for word in words if len(word) > 6]
        metrics["complex_word_ratio"] = len(complex_words) / len(words) if words else 0
        
    except Exception as e:
        logger.warning(f"Complexity analysis failed: {e}")
    
    # Language detection and analysis
    try:
        if nlp:
            doc = nlp(text[:1000])  # Limit for performance
            metrics["language"] = "en"  # Assuming English for now
            metrics["pos_tags"] = {tag: count for tag, count in doc.count_by(spacy.attrs.POS).items()}
        else:
            metrics["language"] = "unknown"
    except Exception as e:
        logger.warning(f"Language analysis failed: {e}")
        metrics["language"] = "unknown"
    
    # Overall quality score
    quality_factors = {
        "readability": min(metrics.get("average_readability", 50) / 100, 1.0),
        "length_adequacy": min(metrics["word_count"] / 500, 1.0),
        "complexity_balance": 1.0 - min(metrics.get("complex_word_ratio", 0.3), 0.5),
        "structure": min(metrics["sentence_count"] / 10, 1.0)
    }
    
    quality_score = sum(quality_factors.values()) / len(quality_factors)
    
    return {
        "quality_score": quality_score,
        "metrics": metrics,
        "quality_factors": quality_factors,
        "analysis_method": "enhanced_textstat_nlp",
        "readability_level": "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "low"
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def enhanced_process_pdf(
    file_content: bytes,
    filename: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Enhanced PDF processing with best libraries and retry logic."""
    options = options or {}
    
    # Extract content
    content_data = await enhanced_extract_content(file_content)
    content_data["filename"] = filename
    
    # Process features in parallel
    tasks = []
    
    if options.get("include_topics", True):
        tasks.append(enhanced_extract_topics(content_data))
    
    if options.get("include_variants", True):
        tasks.append(enhanced_generate_variants(content_data))
    
    if options.get("include_quality", True):
        tasks.append(enhanced_analyze_quality(content_data))
    
    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    processing_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Processing step {i} failed: {result}")
            processing_results.append({"error": str(result)})
        else:
            processing_results.append(result)
    
    return {
        "file_id": hashlib.md5(file_content).hexdigest()[:16],
        "content_data": content_data,
        "processing_results": processing_results,
        "processed_at": time.time(),
        "processing_method": "enhanced_best_libraries",
        "libraries_used": ["pymupdf", "pdfplumber", "spacy", "sentence_transformers", "textstat", "gensim"]
    }


async def enhanced_batch_process(
    files: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 20
) -> Dict[str, Any]:
    """Enhanced batch processing with optimal concurrency."""
    options = options or {}
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_file(file_data):
        async with semaphore:
            try:
                return await enhanced_process_pdf(
                    file_data["content"],
                    file_data["filename"],
                    options
                )
            except Exception as e:
                logger.error(f"Error processing {file_data['filename']}: {e}")
                return {"error": str(e), "filename": file_data["filename"]}
    
    # Process all files
    results = await asyncio.gather(
        *[process_file(file_data) for file_data in files],
        return_exceptions=True
    )
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    return {
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(files) if files else 0,
        "results": results,
        "processing_method": "enhanced_batch_parallel"
    }
