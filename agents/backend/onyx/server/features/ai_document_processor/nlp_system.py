"""
NLP System for AI Document Processor
Real, working Natural Language Processing features for document processing
"""

import asyncio
import logging
import json
import time
import re
import string
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import nltk
import spacy
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import secrets

logger = logging.getLogger(__name__)

class NLPSystem:
    """Real working NLP system for AI document processing"""
    
    def __init__(self):
        self.nlp_models = {}
        self.nlp_pipelines = {}
        self.nlp_corpus = {}
        self.nlp_vocabulary = {}
        self.nlp_embeddings = {}
        self.nlp_similarity = {}
        self.nlp_clusters = {}
        self.nlp_topics = {}
        
        # NLP processing stats
        self.stats = {
            "total_nlp_requests": 0,
            "successful_nlp_requests": 0,
            "failed_nlp_requests": 0,
            "total_tokens_processed": 0,
            "total_sentences_processed": 0,
            "total_documents_processed": 0,
            "vocabulary_size": 0,
            "corpus_size": 0,
            "start_time": time.time()
        }
        
        # Initialize NLP models
        self._initialize_nlp_models()
    
    def _initialize_nlp_models(self):
        """Initialize NLP models and pipelines"""
        try:
            # Initialize spaCy models
            self.nlp_models = {
                "en_core_web_sm": None,  # Will be loaded on demand
                "en_core_web_md": None,  # Will be loaded on demand
                "en_core_web_lg": None  # Will be loaded on demand
            }
            
            # Initialize NLTK components
            self.nlp_pipelines = {
                "tokenizer": None,
                "pos_tagger": None,
                "ner": None,
                "sentiment": None,
                "stemmer": None,
                "lemmatizer": None,
                "stopwords": None
            }
            
            # Initialize NLP corpus
            self.nlp_corpus = {
                "documents": [],
                "sentences": [],
                "tokens": [],
                "metadata": {}
            }
            
            # Initialize vocabulary
            self.nlp_vocabulary = {
                "words": set(),
                "word_frequencies": Counter(),
                "bigrams": Counter(),
                "trigrams": Counter(),
                "pos_tags": Counter(),
                "named_entities": Counter()
            }
            
            # Initialize embeddings
            self.nlp_embeddings = {
                "word_embeddings": {},
                "sentence_embeddings": {},
                "document_embeddings": {},
                "tfidf_matrix": None,
                "vectorizer": None
            }
            
            # Initialize similarity
            self.nlp_similarity = {
                "cosine_similarity": {},
                "jaccard_similarity": {},
                "euclidean_distance": {},
                "manhattan_distance": {}
            }
            
            # Initialize clusters
            self.nlp_clusters = {
                "kmeans": {},
                "hierarchical": {},
                "dbscan": {},
                "clusters": {}
            }
            
            # Initialize topics
            self.nlp_topics = {
                "lda": {},
                "nmf": {},
                "topics": {},
                "topic_distributions": {}
            }
            
            logger.info("NLP system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP system: {e}")
    
    async def load_spacy_model(self, model_name: str = "en_core_web_sm") -> Dict[str, Any]:
        """Load spaCy model"""
        try:
            if model_name not in self.nlp_models or self.nlp_models[model_name] is None:
                import spacy
                self.nlp_models[model_name] = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
            
            return {
                "status": "loaded",
                "model_name": model_name,
                "model_info": {
                    "vocab_size": len(self.nlp_models[model_name].vocab),
                    "pipeline": list(self.nlp_models[model_name].pipe_names)
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            return {"error": str(e)}
    
    async def load_nltk_components(self) -> Dict[str, Any]:
        """Load NLTK components"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize NLTK components
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk.tag import pos_tag
            from nltk.chunk import ne_chunk
            from nltk.sentiment import SentimentIntensityAnalyzer
            from nltk.stem import PorterStemmer, WordNetLemmatizer
            from nltk.corpus import stopwords
            
            self.nlp_pipelines = {
                "tokenizer": word_tokenize,
                "sentence_tokenizer": sent_tokenize,
                "pos_tagger": pos_tag,
                "ner": ne_chunk,
                "sentiment": SentimentIntensityAnalyzer(),
                "stemmer": PorterStemmer(),
                "lemmatizer": WordNetLemmatizer(),
                "stopwords": set(stopwords.words('english'))
            }
            
            return {
                "status": "loaded",
                "components": list(self.nlp_pipelines.keys())
            }
            
        except Exception as e:
            logger.error(f"Error loading NLTK components: {e}")
            return {"error": str(e)}
    
    async def tokenize_text(self, text: str, method: str = "spacy") -> Dict[str, Any]:
        """Tokenize text using different methods"""
        try:
            tokens = []
            
            if method == "spacy":
                # Load spaCy model if not loaded
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                tokens = [token.text for token in doc]
                
            elif method == "nltk":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["tokenizer"] is None:
                    await self.load_nltk_components()
                
                tokens = self.nlp_pipelines["tokenizer"](text)
                
            elif method == "regex":
                # Simple regex tokenization
                tokens = re.findall(r'\b\w+\b', text.lower())
            
            # Update vocabulary
            self.nlp_vocabulary["words"].update(tokens)
            self.nlp_vocabulary["word_frequencies"].update(tokens)
            
            # Update stats
            self.stats["total_tokens_processed"] += len(tokens)
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "tokens": tokens,
                "token_count": len(tokens),
                "unique_tokens": len(set(tokens))
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error tokenizing text: {e}")
            return {"error": str(e)}
    
    async def sentence_segmentation(self, text: str, method: str = "spacy") -> Dict[str, Any]:
        """Segment text into sentences"""
        try:
            sentences = []
            
            if method == "spacy":
                # Load spaCy model if not loaded
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                sentences = [sent.text for sent in doc.sents]
                
            elif method == "nltk":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["sentence_tokenizer"] is None:
                    await self.load_nltk_components()
                
                sentences = self.nlp_pipelines["sentence_tokenizer"](text)
                
            elif method == "regex":
                # Simple regex sentence segmentation
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            # Update corpus
            self.nlp_corpus["sentences"].extend(sentences)
            
            # Update stats
            self.stats["total_sentences_processed"] += len(sentences)
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "sentences": sentences,
                "sentence_count": len(sentences),
                "average_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error segmenting sentences: {e}")
            return {"error": str(e)}
    
    async def pos_tagging(self, text: str, method: str = "spacy") -> Dict[str, Any]:
        """Perform part-of-speech tagging"""
        try:
            pos_tags = []
            
            if method == "spacy":
                # Load spaCy model if not loaded
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]
                
            elif method == "nltk":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["tokenizer"] is None or self.nlp_pipelines["pos_tagger"] is None:
                    await self.load_nltk_components()
                
                tokens = self.nlp_pipelines["tokenizer"](text)
                pos_tags = self.nlp_pipelines["pos_tagger"](tokens)
            
            # Update vocabulary with POS tags
            for _, pos, _ in pos_tags:
                self.nlp_vocabulary["pos_tags"][pos] += 1
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "pos_tags": pos_tags,
                "tag_count": len(pos_tags),
                "unique_tags": len(set(tag for _, tag, _ in pos_tags))
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error performing POS tagging: {e}")
            return {"error": str(e)}
    
    async def named_entity_recognition(self, text: str, method: str = "spacy") -> Dict[str, Any]:
        """Perform named entity recognition"""
        try:
            entities = []
            
            if method == "spacy":
                # Load spaCy model if not loaded
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
                
            elif method == "nltk":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["tokenizer"] is None or self.nlp_pipelines["ner"] is None:
                    await self.load_nltk_components()
                
                tokens = self.nlp_pipelines["tokenizer"](text)
                pos_tags = self.nlp_pipelines["pos_tagger"](tokens)
                ner_tree = self.nlp_pipelines["ner"](pos_tags)
                
                entities = []
                for chunk in ner_tree:
                    if hasattr(chunk, 'label'):
                        entities.append((chunk.leaves()[0][0], chunk.label(), 0, 0))
            
            # Update vocabulary with named entities
            for entity, label, _, _ in entities:
                self.nlp_vocabulary["named_entities"][label] += 1
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "entities": entities,
                "entity_count": len(entities),
                "unique_entity_types": len(set(label for _, label, _, _ in entities))
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error performing NER: {e}")
            return {"error": str(e)}
    
    async def sentiment_analysis(self, text: str, method: str = "nltk") -> Dict[str, Any]:
        """Perform sentiment analysis"""
        try:
            sentiment_scores = {}
            
            if method == "nltk":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["sentiment"] is None:
                    await self.load_nltk_components()
                
                sentiment_scores = self.nlp_pipelines["sentiment"].polarity_scores(text)
                
            elif method == "spacy":
                # Load spaCy model if not loaded
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                # Simple sentiment analysis using spaCy
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                negative_words = ["bad", "terrible", "awful", "horrible", "disgusting", "hate"]
                
                positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
                negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
                
                total_sentiment = positive_count - negative_count
                sentiment_scores = {
                    "compound": total_sentiment / max(len(doc), 1),
                    "pos": positive_count / max(len(doc), 1),
                    "neu": 1 - (positive_count + negative_count) / max(len(doc), 1),
                    "neg": negative_count / max(len(doc), 1)
                }
            
            # Determine sentiment label
            if sentiment_scores["compound"] >= 0.05:
                sentiment_label = "positive"
            elif sentiment_scores["compound"] <= -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "sentiment_scores": sentiment_scores,
                "sentiment_label": sentiment_label,
                "confidence": abs(sentiment_scores["compound"])
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error performing sentiment analysis: {e}")
            return {"error": str(e)}
    
    async def text_preprocessing(self, text: str, steps: List[str] = None) -> Dict[str, Any]:
        """Comprehensive text preprocessing"""
        try:
            if steps is None:
                steps = ["lowercase", "remove_punctuation", "remove_stopwords", "lemmatize"]
            
            processed_text = text
            preprocessing_steps = []
            
            # Load NLTK components if needed
            if any(step in ["remove_stopwords", "lemmatize", "stem"] for step in steps):
                if self.nlp_pipelines["stopwords"] is None:
                    await self.load_nltk_components()
            
            # Load spaCy model if needed
            if "lemmatize" in steps:
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_spacy_model("en_core_web_sm")
            
            # Apply preprocessing steps
            for step in steps:
                if step == "lowercase":
                    processed_text = processed_text.lower()
                    preprocessing_steps.append("lowercase")
                
                elif step == "remove_punctuation":
                    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
                    preprocessing_steps.append("remove_punctuation")
                
                elif step == "remove_stopwords":
                    tokens = processed_text.split()
                    processed_text = " ".join([token for token in tokens if token not in self.nlp_pipelines["stopwords"]])
                    preprocessing_steps.append("remove_stopwords")
                
                elif step == "lemmatize":
                    doc = self.nlp_models["en_core_web_sm"](processed_text)
                    processed_text = " ".join([token.lemma_ for token in doc])
                    preprocessing_steps.append("lemmatize")
                
                elif step == "stem":
                    tokens = processed_text.split()
                    processed_text = " ".join([self.nlp_pipelines["stemmer"].stem(token) for token in tokens])
                    preprocessing_steps.append("stem")
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "original_text": text,
                "processed_text": processed_text,
                "preprocessing_steps": preprocessing_steps,
                "original_length": len(text),
                "processed_length": len(processed_text)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error preprocessing text: {e}")
            return {"error": str(e)}
    
    async def extract_keywords(self, text: str, method: str = "tfidf", top_k: int = 10) -> Dict[str, Any]:
        """Extract keywords from text"""
        try:
            keywords = []
            
            if method == "tfidf":
                # Use TF-IDF for keyword extraction
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(
                        max_features=1000,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                
                # Fit and transform the text
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform([text])
                feature_names = self.nlp_embeddings["vectorizer"].get_feature_names_out()
                
                # Get top keywords
                tfidf_scores = tfidf_matrix.toarray()[0]
                keyword_scores = list(zip(feature_names, tfidf_scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                
                keywords = [{"keyword": kw, "score": score} for kw, score in keyword_scores[:top_k]]
                
            elif method == "frequency":
                # Use word frequency for keyword extraction
                tokens = re.findall(r'\b\w+\b', text.lower())
                word_freq = Counter(tokens)
                
                # Remove stopwords
                if self.nlp_pipelines["stopwords"] is None:
                    await self.load_nltk_components()
                
                filtered_words = {word: freq for word, freq in word_freq.items() 
                                if word not in self.nlp_pipelines["stopwords"] and len(word) > 2}
                
                keywords = [{"keyword": word, "score": freq} for word, freq in 
                          sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "keywords": keywords,
                "keyword_count": len(keywords)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error extracting keywords: {e}")
            return {"error": str(e)}
    
    async def calculate_similarity(self, text1: str, text2: str, method: str = "cosine") -> Dict[str, Any]:
        """Calculate similarity between two texts"""
        try:
            similarity_score = 0.0
            
            if method == "cosine":
                # Use TF-IDF and cosine similarity
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(stop_words='english')
                
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform([text1, text2])
                similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                
            elif method == "jaccard":
                # Use Jaccard similarity
                set1 = set(re.findall(r'\b\w+\b', text1.lower()))
                set2 = set(re.findall(r'\b\w+\b', text2.lower()))
                
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                similarity_score = intersection / union if union > 0 else 0
                
            elif method == "euclidean":
                # Use Euclidean distance (convert to similarity)
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(stop_words='english')
                
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform([text1, text2])
                distance = np.linalg.norm(tfidf_matrix[0].toarray() - tfidf_matrix[1].toarray())
                similarity_score = 1 / (1 + distance)
            
            # Store similarity
            similarity_key = f"{hash(text1)}_{hash(text2)}"
            self.nlp_similarity[method][similarity_key] = similarity_score
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "similarity_score": similarity_score,
                "text1_length": len(text1),
                "text2_length": len(text2)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error calculating similarity: {e}")
            return {"error": str(e)}
    
    async def topic_modeling(self, texts: List[str], method: str = "lda", num_topics: int = 5) -> Dict[str, Any]:
        """Perform topic modeling on a collection of texts"""
        try:
            topics = []
            
            if method == "lda":
                # Use Latent Dirichlet Allocation
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(
                        max_features=1000,
                        stop_words='english',
                        ngram_range=(1, 2)
                    )
                
                # Prepare texts
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform(texts)
                feature_names = self.nlp_embeddings["vectorizer"].get_feature_names_out()
                
                # Fit LDA model
                lda = LatentDirichletAllocation(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=100
                )
                lda.fit(tfidf_matrix)
                
                # Extract topics
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        "topic_id": topic_idx,
                        "top_words": top_words,
                        "word_weights": topic[top_words_idx].tolist()
                    })
                
                # Store LDA model
                self.nlp_topics["lda"] = lda
                
            elif method == "kmeans":
                # Use K-means clustering
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(
                        max_features=1000,
                        stop_words='english'
                    )
                
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform(texts)
                
                # Fit K-means
                kmeans = KMeans(n_clusters=num_topics, random_state=42)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                # Extract topics from clusters
                feature_names = self.nlp_embeddings["vectorizer"].get_feature_names_out()
                for cluster_idx in range(num_topics):
                    cluster_center = kmeans.cluster_centers_[cluster_idx]
                    top_words_idx = cluster_center.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        "topic_id": cluster_idx,
                        "top_words": top_words,
                        "word_weights": cluster_center[top_words_idx].tolist()
                    })
                
                # Store K-means model
                self.nlp_clusters["kmeans"] = kmeans
            
            # Store topics
            self.nlp_topics["topics"] = topics
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "topics": topics,
                "num_topics": num_topics,
                "num_documents": len(texts)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error performing topic modeling: {e}")
            return {"error": str(e)}
    
    async def text_classification(self, text: str, categories: List[str], method: str = "naive_bayes") -> Dict[str, Any]:
        """Classify text into categories"""
        try:
            classification_result = {}
            
            if method == "naive_bayes":
                # Simple Naive Bayes classification based on keywords
                text_lower = text.lower()
                
                # Define category keywords
                category_keywords = {
                    "technology": ["computer", "software", "hardware", "digital", "tech", "programming", "code"],
                    "business": ["company", "market", "sales", "profit", "revenue", "business", "corporate"],
                    "science": ["research", "study", "experiment", "scientific", "data", "analysis", "hypothesis"],
                    "sports": ["game", "player", "team", "match", "sport", "athlete", "competition"],
                    "politics": ["government", "policy", "election", "vote", "political", "democracy", "law"]
                }
                
                # Calculate scores for each category
                category_scores = {}
                for category, keywords in category_keywords.items():
                    if category in categories:
                        score = sum(1 for keyword in keywords if keyword in text_lower)
                        category_scores[category] = score
                
                # Find best category
                if category_scores:
                    best_category = max(category_scores, key=category_scores.get)
                    confidence = category_scores[best_category] / sum(category_scores.values()) if sum(category_scores.values()) > 0 else 0
                else:
                    best_category = "unknown"
                    confidence = 0
                
                classification_result = {
                    "predicted_category": best_category,
                    "confidence": confidence,
                    "category_scores": category_scores
                }
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "classification": classification_result,
                "available_categories": categories
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error classifying text: {e}")
            return {"error": str(e)}
    
    async def text_summarization(self, text: str, method: str = "extractive", max_sentences: int = 3) -> Dict[str, Any]:
        """Summarize text using different methods"""
        try:
            summary = ""
            
            if method == "extractive":
                # Extractive summarization using sentence scoring
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) <= max_sentences:
                    summary = text
                else:
                    # Score sentences based on word frequency
                    word_freq = Counter(re.findall(r'\b\w+\b', text.lower()))
                    
                    sentence_scores = []
                    for sentence in sentences:
                        words = re.findall(r'\b\w+\b', sentence.lower())
                        score = sum(word_freq[word] for word in words)
                        sentence_scores.append((sentence, score))
                    
                    # Select top sentences
                    sentence_scores.sort(key=lambda x: x[1], reverse=True)
                    top_sentences = [sent for sent, _ in sentence_scores[:max_sentences]]
                    summary = ". ".join(top_sentences) + "."
            
            elif method == "abstractive":
                # Simple abstractive summarization (placeholder)
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) <= max_sentences:
                    summary = text
                else:
                    # Take first and last sentences
                    summary = sentences[0] + ". " + sentences[-1] + "."
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text) if len(text) > 0 else 0
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error summarizing text: {e}")
            return {"error": str(e)}
    
    def get_nlp_models(self) -> Dict[str, Any]:
        """Get all NLP models"""
        return {
            "spacy_models": {k: v is not None for k, v in self.nlp_models.items()},
            "nltk_components": {k: v is not None for k, v in self.nlp_pipelines.items()},
            "model_count": len([m for m in self.nlp_models.values() if m is not None])
        }
    
    def get_nlp_corpus(self) -> Dict[str, Any]:
        """Get NLP corpus information"""
        return {
            "corpus": self.nlp_corpus,
            "document_count": len(self.nlp_corpus["documents"]),
            "sentence_count": len(self.nlp_corpus["sentences"]),
            "token_count": len(self.nlp_corpus["tokens"])
        }
    
    def get_nlp_vocabulary(self) -> Dict[str, Any]:
        """Get NLP vocabulary information"""
        return {
            "vocabulary": self.nlp_vocabulary,
            "vocabulary_size": len(self.nlp_vocabulary["words"]),
            "word_frequencies": dict(self.nlp_vocabulary["word_frequencies"].most_common(20)),
            "pos_tags": dict(self.nlp_vocabulary["pos_tags"].most_common(10)),
            "named_entities": dict(self.nlp_vocabulary["named_entities"].most_common(10))
        }
    
    def get_nlp_embeddings(self) -> Dict[str, Any]:
        """Get NLP embeddings information"""
        return {
            "embeddings": self.nlp_embeddings,
            "word_embeddings_count": len(self.nlp_embeddings["word_embeddings"]),
            "sentence_embeddings_count": len(self.nlp_embeddings["sentence_embeddings"]),
            "document_embeddings_count": len(self.nlp_embeddings["document_embeddings"])
        }
    
    def get_nlp_similarity(self) -> Dict[str, Any]:
        """Get NLP similarity information"""
        return {
            "similarity": self.nlp_similarity,
            "cosine_similarities": len(self.nlp_similarity["cosine_similarity"]),
            "jaccard_similarities": len(self.nlp_similarity["jaccard_similarity"]),
            "euclidean_distances": len(self.nlp_similarity["euclidean_distance"]),
            "manhattan_distances": len(self.nlp_similarity["manhattan_distance"])
        }
    
    def get_nlp_clusters(self) -> Dict[str, Any]:
        """Get NLP clusters information"""
        return {
            "clusters": self.nlp_clusters,
            "kmeans_clusters": len(self.nlp_clusters["kmeans"]),
            "hierarchical_clusters": len(self.nlp_clusters["hierarchical"]),
            "dbscan_clusters": len(self.nlp_clusters["dbscan"])
        }
    
    def get_nlp_topics(self) -> Dict[str, Any]:
        """Get NLP topics information"""
        return {
            "topics": self.nlp_topics,
            "lda_topics": len(self.nlp_topics["lda"]),
            "nmf_topics": len(self.nlp_topics["nmf"]),
            "topic_count": len(self.nlp_topics["topics"])
        }
    
    def get_nlp_stats(self) -> Dict[str, Any]:
        """Get NLP processing statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "success_rate": (self.stats["successful_nlp_requests"] / self.stats["total_nlp_requests"] * 100) if self.stats["total_nlp_requests"] > 0 else 0,
            "average_tokens_per_request": self.stats["total_tokens_processed"] / self.stats["total_nlp_requests"] if self.stats["total_nlp_requests"] > 0 else 0,
            "average_sentences_per_request": self.stats["total_sentences_processed"] / self.stats["total_nlp_requests"] if self.stats["total_nlp_requests"] > 0 else 0
        }

# Global instance
nlp_system = NLPSystem()












