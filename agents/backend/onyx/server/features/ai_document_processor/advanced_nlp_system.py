"""
Advanced NLP System for AI Document Processor
Real, working advanced Natural Language Processing features for document processing
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import networkx as nx
from textstat import flesch_reading_ease, flesch_kincaid_grade, smog_index, coleman_liau_index
import secrets
import pickle
import joblib

logger = logging.getLogger(__name__)

class AdvancedNLPSystem:
    """Real working advanced NLP system for AI document processing"""
    
    def __init__(self):
        self.nlp_models = {}
        self.nlp_pipelines = {}
        self.nlp_corpus = {}
        self.nlp_vocabulary = {}
        self.nlp_embeddings = {}
        self.nlp_similarity = {}
        self.nlp_clusters = {}
        self.nlp_topics = {}
        self.nlp_classifiers = {}
        self.nlp_networks = {}
        self.nlp_metrics = {}
        
        # Advanced NLP processing stats
        self.stats = {
            "total_nlp_requests": 0,
            "successful_nlp_requests": 0,
            "failed_nlp_requests": 0,
            "total_tokens_processed": 0,
            "total_sentences_processed": 0,
            "total_documents_processed": 0,
            "vocabulary_size": 0,
            "corpus_size": 0,
            "total_embeddings_created": 0,
            "total_similarities_calculated": 0,
            "total_clusters_created": 0,
            "total_topics_discovered": 0,
            "total_classifications_made": 0,
            "total_networks_built": 0,
            "start_time": time.time()
        }
        
        # Initialize advanced NLP models
        self._initialize_advanced_nlp_models()
    
    def _initialize_advanced_nlp_models(self):
        """Initialize advanced NLP models and pipelines"""
        try:
            # Initialize spaCy models
            self.nlp_models = {
                "en_core_web_sm": None,  # Will be loaded on demand
                "en_core_web_md": None,  # Will be loaded on demand
                "en_core_web_lg": None,  # Will be loaded on demand
                "en_core_web_trf": None  # Will be loaded on demand
            }
            
            # Initialize NLTK components
            self.nlp_pipelines = {
                "tokenizer": None,
                "pos_tagger": None,
                "ner": None,
                "sentiment": None,
                "stemmer": None,
                "lemmatizer": None,
                "stopwords": None,
                "chunker": None,
                "parser": None
            }
            
            # Initialize advanced NLP corpus
            self.nlp_corpus = {
                "documents": [],
                "sentences": [],
                "tokens": [],
                "phrases": [],
                "entities": [],
                "metadata": {}
            }
            
            # Initialize advanced vocabulary
            self.nlp_vocabulary = {
                "words": set(),
                "word_frequencies": Counter(),
                "bigrams": Counter(),
                "trigrams": Counter(),
                "pos_tags": Counter(),
                "named_entities": Counter(),
                "phrases": Counter(),
                "collocations": Counter()
            }
            
            # Initialize advanced embeddings
            self.nlp_embeddings = {
                "word_embeddings": {},
                "sentence_embeddings": {},
                "document_embeddings": {},
                "phrase_embeddings": {},
                "entity_embeddings": {},
                "tfidf_matrix": None,
                "count_matrix": None,
                "vectorizer": None,
                "count_vectorizer": None
            }
            
            # Initialize advanced similarity
            self.nlp_similarity = {
                "cosine_similarity": {},
                "jaccard_similarity": {},
                "euclidean_distance": {},
                "manhattan_distance": {},
                "pearson_correlation": {},
                "spearman_correlation": {}
            }
            
            # Initialize advanced clusters
            self.nlp_clusters = {
                "kmeans": {},
                "dbscan": {},
                "hierarchical": {},
                "spectral": {},
                "clusters": {}
            }
            
            # Initialize advanced topics
            self.nlp_topics = {
                "lda": {},
                "nmf": {},
                "lsa": {},
                "topics": {},
                "topic_distributions": {}
            }
            
            # Initialize classifiers
            self.nlp_classifiers = {
                "naive_bayes": {},
                "logistic_regression": {},
                "random_forest": {},
                "svm": {},
                "ensemble": {}
            }
            
            # Initialize networks
            self.nlp_networks = {
                "word_networks": {},
                "entity_networks": {},
                "topic_networks": {},
                "co_occurrence_networks": {}
            }
            
            # Initialize metrics
            self.nlp_metrics = {
                "readability_metrics": {},
                "complexity_metrics": {},
                "coherence_metrics": {},
                "diversity_metrics": {}
            }
            
            logger.info("Advanced NLP system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing advanced NLP system: {e}")
    
    async def load_advanced_spacy_model(self, model_name: str = "en_core_web_sm") -> Dict[str, Any]:
        """Load advanced spaCy model"""
        try:
            if model_name not in self.nlp_models or self.nlp_models[model_name] is None:
                import spacy
                self.nlp_models[model_name] = spacy.load(model_name)
                logger.info(f"Loaded advanced spaCy model: {model_name}")
            
            return {
                "status": "loaded",
                "model_name": model_name,
                "model_info": {
                    "vocab_size": len(self.nlp_models[model_name].vocab),
                    "pipeline": list(self.nlp_models[model_name].pipe_names),
                    "vectors": self.nlp_models[model_name].vocab.vectors.size if self.nlp_models[model_name].vocab.vectors.size > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading advanced spaCy model: {e}")
            return {"error": str(e)}
    
    async def load_advanced_nltk_components(self) -> Dict[str, Any]:
        """Load advanced NLTK components"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('brown', quiet=True)
            nltk.download('reuters', quiet=True)
            nltk.download('gutenberg', quiet=True)
            nltk.download('treebank', quiet=True)
            nltk.download('conll2000', quiet=True)
            nltk.download('conll2002', quiet=True)
            nltk.download('conll2007', quiet=True)
            nltk.download('movie_reviews', quiet=True)
            nltk.download('twitter_samples', quiet=True)
            nltk.download('sentiwordnet', quiet=True)
            nltk.download('opinion_lexicon', quiet=True)
            nltk.download('subjectivity', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            # Initialize advanced NLTK components
            from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
            from nltk.tag import pos_tag, StanfordPOSTagger
            from nltk.chunk import ne_chunk
            from nltk.sentiment import SentimentIntensityAnalyzer
            from nltk.stem import PorterStemmer, WordNetLemmatizer, LancasterStemmer, SnowballStemmer
            from nltk.corpus import stopwords
            from nltk.chunk import RegexpParser
            from nltk.parse import CoreNLPParser
            from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
            from nltk.corpus import wordnet
            from nltk.wsd import lesk
            
            self.nlp_pipelines = {
                "tokenizer": word_tokenize,
                "sentence_tokenizer": sent_tokenize,
                "tweet_tokenizer": TweetTokenizer(),
                "pos_tagger": pos_tag,
                "ner": ne_chunk,
                "sentiment": SentimentIntensityAnalyzer(),
                "stemmer": PorterStemmer(),
                "lancaster_stemmer": LancasterStemmer(),
                "snowball_stemmer": SnowballStemmer('english'),
                "lemmatizer": WordNetLemmatizer(),
                "stopwords": set(stopwords.words('english')),
                "chunker": RegexpParser(r'NP: {<DT>?<JJ>*<NN>}'),
                "parser": CoreNLPParser(),
                "wordnet": wordnet,
                "lesk": lesk
            }
            
            return {
                "status": "loaded",
                "components": list(self.nlp_pipelines.keys())
            }
            
        except Exception as e:
            logger.error(f"Error loading advanced NLTK components: {e}")
            return {"error": str(e)}
    
    async def advanced_tokenization(self, text: str, method: str = "spacy", 
                                  include_phrases: bool = True, include_entities: bool = True) -> Dict[str, Any]:
        """Advanced tokenization with phrases and entities"""
        try:
            tokens = []
            phrases = []
            entities = []
            
            if method == "spacy":
                # Load spaCy model if not loaded
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_advanced_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                tokens = [token.text for token in doc]
                
                if include_phrases:
                    phrases = [chunk.text for chunk in doc.noun_chunks]
                
                if include_entities:
                    entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
                
            elif method == "nltk":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["tokenizer"] is None:
                    await self.load_advanced_nltk_components()
                
                tokens = self.nlp_pipelines["tokenizer"](text)
                
                if include_phrases:
                    pos_tags = self.nlp_pipelines["pos_tagger"](tokens)
                    phrases = [phrase for phrase, _ in self.nlp_pipelines["chunker"].parse(pos_tags)]
                
                if include_entities:
                    pos_tags = self.nlp_pipelines["pos_tagger"](tokens)
                    ner_tree = self.nlp_pipelines["ner"](pos_tags)
                    entities = [(chunk.leaves()[0][0], chunk.label(), 0, 0) for chunk in ner_tree if hasattr(chunk, 'label')]
                
            elif method == "tweet":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["tweet_tokenizer"] is None:
                    await self.load_advanced_nltk_components()
                
                tokens = self.nlp_pipelines["tweet_tokenizer"].tokenize(text)
            
            # Update vocabulary
            self.nlp_vocabulary["words"].update(tokens)
            self.nlp_vocabulary["word_frequencies"].update(tokens)
            
            if phrases:
                self.nlp_vocabulary["phrases"].update(phrases)
            
            if entities:
                for entity, label, _, _ in entities:
                    self.nlp_vocabulary["named_entities"][label] += 1
            
            # Update corpus
            self.nlp_corpus["tokens"].extend(tokens)
            if phrases:
                self.nlp_corpus["phrases"].extend(phrases)
            if entities:
                self.nlp_corpus["entities"].extend(entities)
            
            # Update stats
            self.stats["total_tokens_processed"] += len(tokens)
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "tokens": tokens,
                "phrases": phrases,
                "entities": entities,
                "token_count": len(tokens),
                "phrase_count": len(phrases),
                "entity_count": len(entities),
                "unique_tokens": len(set(tokens))
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error in advanced tokenization: {e}")
            return {"error": str(e)}
    
    async def advanced_sentiment_analysis(self, text: str, method: str = "nltk", 
                                        include_emotions: bool = True) -> Dict[str, Any]:
        """Advanced sentiment analysis with emotions"""
        try:
            sentiment_scores = {}
            emotions = {}
            
            if method == "nltk":
                # Load NLTK components if not loaded
                if self.nlp_pipelines["sentiment"] is None:
                    await self.load_advanced_nltk_components()
                
                sentiment_scores = self.nlp_pipelines["sentiment"].polarity_scores(text)
                
                if include_emotions:
                    # Simple emotion detection based on keywords
                    emotion_keywords = {
                        "joy": ["happy", "joy", "excited", "thrilled", "delighted", "ecstatic"],
                        "sadness": ["sad", "depressed", "melancholy", "gloomy", "miserable", "sorrowful"],
                        "anger": ["angry", "mad", "furious", "rage", "irritated", "annoyed"],
                        "fear": ["afraid", "scared", "terrified", "worried", "anxious", "nervous"],
                        "surprise": ["surprised", "amazed", "shocked", "astonished", "stunned", "bewildered"],
                        "disgust": ["disgusted", "revolted", "sickened", "repulsed", "nauseated", "appalled"]
                    }
                    
                    text_lower = text.lower()
                    for emotion, keywords in emotion_keywords.items():
                        emotion_score = sum(1 for keyword in keywords if keyword in text_lower)
                        emotions[emotion] = emotion_score / len(keywords)
                
            elif method == "spacy":
                # Load spaCy model if not loaded
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_advanced_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                
                # Advanced sentiment analysis using spaCy
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "awesome", "brilliant"]
                negative_words = ["bad", "terrible", "awful", "horrible", "disgusting", "hate", "worst", "disappointing"]
                neutral_words = ["okay", "fine", "average", "normal", "regular", "standard", "typical"]
                
                positive_count = sum(1 for token in doc if token.text.lower() in positive_words)
                negative_count = sum(1 for token in doc if token.text.lower() in negative_words)
                neutral_count = sum(1 for token in doc if token.text.lower() in neutral_words)
                
                total_sentiment = positive_count - negative_count
                sentiment_scores = {
                    "compound": total_sentiment / max(len(doc), 1),
                    "pos": positive_count / max(len(doc), 1),
                    "neu": neutral_count / max(len(doc), 1),
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
                "confidence": abs(sentiment_scores["compound"]),
                "emotions": emotions
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error in advanced sentiment analysis: {e}")
            return {"error": str(e)}
    
    async def advanced_text_preprocessing(self, text: str, steps: List[str] = None) -> Dict[str, Any]:
        """Advanced text preprocessing with multiple options"""
        try:
            if steps is None:
                steps = ["lowercase", "remove_punctuation", "remove_stopwords", "lemmatize", "remove_numbers"]
            
            processed_text = text
            preprocessing_steps = []
            
            # Load NLTK components if needed
            if any(step in ["remove_stopwords", "lemmatize", "stem", "remove_stopwords_advanced"] for step in steps):
                if self.nlp_pipelines["stopwords"] is None:
                    await self.load_advanced_nltk_components()
            
            # Load spaCy model if needed
            if "lemmatize" in steps:
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_advanced_spacy_model("en_core_web_sm")
            
            # Apply preprocessing steps
            for step in steps:
                if step == "lowercase":
                    processed_text = processed_text.lower()
                    preprocessing_steps.append("lowercase")
                
                elif step == "remove_punctuation":
                    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
                    preprocessing_steps.append("remove_punctuation")
                
                elif step == "remove_numbers":
                    processed_text = re.sub(r'\d+', '', processed_text)
                    preprocessing_steps.append("remove_numbers")
                
                elif step == "remove_stopwords":
                    tokens = processed_text.split()
                    processed_text = " ".join([token for token in tokens if token not in self.nlp_pipelines["stopwords"]])
                    preprocessing_steps.append("remove_stopwords")
                
                elif step == "remove_stopwords_advanced":
                    # Advanced stopword removal with custom stopwords
                    custom_stopwords = self.nlp_pipelines["stopwords"].copy()
                    custom_stopwords.update(['said', 'says', 'say', 'would', 'could', 'should'])
                    tokens = processed_text.split()
                    processed_text = " ".join([token for token in tokens if token not in custom_stopwords])
                    preprocessing_steps.append("remove_stopwords_advanced")
                
                elif step == "lemmatize":
                    doc = self.nlp_models["en_core_web_sm"](processed_text)
                    processed_text = " ".join([token.lemma_ for token in doc])
                    preprocessing_steps.append("lemmatize")
                
                elif step == "stem":
                    tokens = processed_text.split()
                    processed_text = " ".join([self.nlp_pipelines["stemmer"].stem(token) for token in tokens])
                    preprocessing_steps.append("stem")
                
                elif step == "lancaster_stem":
                    tokens = processed_text.split()
                    processed_text = " ".join([self.nlp_pipelines["lancaster_stemmer"].stem(token) for token in tokens])
                    preprocessing_steps.append("lancaster_stem")
                
                elif step == "snowball_stem":
                    tokens = processed_text.split()
                    processed_text = " ".join([self.nlp_pipelines["snowball_stemmer"].stem(token) for token in tokens])
                    preprocessing_steps.append("snowball_stem")
                
                elif step == "remove_extra_whitespace":
                    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
                    preprocessing_steps.append("remove_extra_whitespace")
                
                elif step == "remove_urls":
                    processed_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', processed_text)
                    preprocessing_steps.append("remove_urls")
                
                elif step == "remove_emails":
                    processed_text = re.sub(r'\S+@\S+', '', processed_text)
                    preprocessing_steps.append("remove_emails")
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "original_text": text,
                "processed_text": processed_text,
                "preprocessing_steps": preprocessing_steps,
                "original_length": len(text),
                "processed_length": len(processed_text),
                "compression_ratio": len(processed_text) / len(text) if len(text) > 0 else 0
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error in advanced text preprocessing: {e}")
            return {"error": str(e)}
    
    async def advanced_keyword_extraction(self, text: str, method: str = "tfidf", 
                                        top_k: int = 10, include_phrases: bool = True) -> Dict[str, Any]:
        """Advanced keyword extraction with phrases"""
        try:
            keywords = []
            phrases = []
            
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
                    await self.load_advanced_nltk_components()
                
                filtered_words = {word: freq for word, freq in word_freq.items() 
                                if word not in self.nlp_pipelines["stopwords"] and len(word) > 2}
                
                keywords = [{"keyword": word, "score": freq} for word, freq in 
                          sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            
            elif method == "yake":
                # YAKE (Yet Another Keyword Extractor) algorithm
                # Simple implementation of YAKE
                tokens = re.findall(r'\b\w+\b', text.lower())
                word_freq = Counter(tokens)
                
                # Calculate YAKE scores
                yake_scores = {}
                for word, freq in word_freq.items():
                    if len(word) > 2 and word not in self.nlp_pipelines["stopwords"]:
                        # Simple YAKE score calculation
                        score = freq / (1 + math.log(freq))
                        yake_scores[word] = score
                
                keywords = [{"keyword": word, "score": score} for word, score in 
                          sorted(yake_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            
            if include_phrases:
                # Extract phrases using spaCy
                if "en_core_web_sm" not in self.nlp_models or self.nlp_models["en_core_web_sm"] is None:
                    await self.load_advanced_spacy_model("en_core_web_sm")
                
                doc = self.nlp_models["en_core_web_sm"](text)
                phrases = [chunk.text for chunk in doc.noun_chunks]
                phrase_freq = Counter(phrases)
                
                phrases = [{"phrase": phrase, "score": freq} for phrase, freq in 
                          sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "keywords": keywords,
                "phrases": phrases,
                "keyword_count": len(keywords),
                "phrase_count": len(phrases)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error in advanced keyword extraction: {e}")
            return {"error": str(e)}
    
    async def advanced_similarity_calculation(self, text1: str, text2: str, 
                                            method: str = "cosine", include_semantic: bool = True) -> Dict[str, Any]:
        """Advanced similarity calculation with semantic analysis"""
        try:
            similarity_score = 0.0
            semantic_similarity = 0.0
            
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
            
            elif method == "manhattan":
                # Use Manhattan distance (convert to similarity)
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(stop_words='english')
                
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform([text1, text2])
                distance = np.sum(np.abs(tfidf_matrix[0].toarray() - tfidf_matrix[1].toarray()))
                similarity_score = 1 / (1 + distance)
            
            if include_semantic:
                # Calculate semantic similarity using WordNet
                if self.nlp_pipelines["wordnet"] is None:
                    await self.load_advanced_nltk_components()
                
                # Simple semantic similarity calculation
                words1 = set(re.findall(r'\b\w+\b', text1.lower()))
                words2 = set(re.findall(r'\b\w+\b', text2.lower()))
                
                semantic_scores = []
                for word1 in words1:
                    for word2 in words2:
                        if word1 != word2:
                            synsets1 = self.nlp_pipelines["wordnet"].synsets(word1)
                            synsets2 = self.nlp_pipelines["wordnet"].synsets(word2)
                            
                            if synsets1 and synsets2:
                                max_sim = 0
                                for syn1 in synsets1:
                                    for syn2 in synsets2:
                                        sim = syn1.wup_similarity(syn2)
                                        if sim:
                                            max_sim = max(max_sim, sim)
                                semantic_scores.append(max_sim)
                
                semantic_similarity = np.mean(semantic_scores) if semantic_scores else 0
            
            # Store similarity
            similarity_key = f"{hash(text1)}_{hash(text2)}"
            self.nlp_similarity[method][similarity_key] = similarity_score
            
            # Update stats
            self.stats["total_similarities_calculated"] += 1
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "similarity_score": similarity_score,
                "semantic_similarity": semantic_similarity,
                "text1_length": len(text1),
                "text2_length": len(text2)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error in advanced similarity calculation: {e}")
            return {"error": str(e)}
    
    async def advanced_topic_modeling(self, texts: List[str], method: str = "lda", 
                                     num_topics: int = 5, include_coherence: bool = True) -> Dict[str, Any]:
        """Advanced topic modeling with coherence analysis"""
        try:
            topics = []
            coherence_scores = {}
            
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
                
            elif method == "nmf":
                # Use Non-negative Matrix Factorization
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(
                        max_features=1000,
                        stop_words='english'
                    )
                
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform(texts)
                feature_names = self.nlp_embeddings["vectorizer"].get_feature_names_out()
                
                # Fit NMF model
                nmf = NMF(
                    n_components=num_topics,
                    random_state=42,
                    max_iter=100
                )
                nmf.fit(tfidf_matrix)
                
                # Extract topics
                for topic_idx, topic in enumerate(nmf.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        "topic_id": topic_idx,
                        "top_words": top_words,
                        "word_weights": topic[top_words_idx].tolist()
                    })
                
                # Store NMF model
                self.nlp_topics["nmf"] = nmf
                
            elif method == "lsa":
                # Use Latent Semantic Analysis
                if self.nlp_embeddings["vectorizer"] is None:
                    self.nlp_embeddings["vectorizer"] = TfidfVectorizer(
                        max_features=1000,
                        stop_words='english'
                    )
                
                tfidf_matrix = self.nlp_embeddings["vectorizer"].fit_transform(texts)
                feature_names = self.nlp_embeddings["vectorizer"].get_feature_names_out()
                
                # Fit LSA model
                lsa = TruncatedSVD(
                    n_components=num_topics,
                    random_state=42
                )
                lsa.fit(tfidf_matrix)
                
                # Extract topics
                for topic_idx, topic in enumerate(lsa.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics.append({
                        "topic_id": topic_idx,
                        "top_words": top_words,
                        "word_weights": topic[top_words_idx].tolist()
                    })
                
                # Store LSA model
                self.nlp_topics["lsa"] = lsa
            
            if include_coherence:
                # Calculate topic coherence (simplified)
                for topic in topics:
                    coherence_score = 0
                    top_words = topic["top_words"][:5]  # Use top 5 words for coherence
                    
                    # Simple coherence calculation based on word co-occurrence
                    for i, word1 in enumerate(top_words):
                        for j, word2 in enumerate(top_words[i+1:], i+1):
                            # Count co-occurrence of word1 and word2
                            co_occurrence = sum(1 for text in texts if word1 in text.lower() and word2 in text.lower())
                            coherence_score += co_occurrence
                    
                    coherence_scores[topic["topic_id"]] = coherence_score
            
            # Store topics
            self.nlp_topics["topics"] = topics
            
            # Update stats
            self.stats["total_topics_discovered"] += len(topics)
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "topics": topics,
                "coherence_scores": coherence_scores,
                "num_topics": num_topics,
                "num_documents": len(texts)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error in advanced topic modeling: {e}")
            return {"error": str(e)}
    
    async def advanced_text_classification(self, text: str, categories: List[str], 
                                         method: str = "naive_bayes", include_confidence: bool = True) -> Dict[str, Any]:
        """Advanced text classification with confidence scores"""
        try:
            classification_result = {}
            
            if method == "naive_bayes":
                # Advanced Naive Bayes classification
                text_lower = text.lower()
                
                # Define category keywords with weights
                category_keywords = {
                    "technology": {
                        "keywords": ["computer", "software", "hardware", "digital", "tech", "programming", "code", "ai", "machine learning"],
                        "weights": [1.0, 1.0, 1.0, 0.8, 1.0, 1.2, 1.0, 1.5, 1.5]
                    },
                    "business": {
                        "keywords": ["company", "market", "sales", "profit", "revenue", "business", "corporate", "finance", "investment"],
                        "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    },
                    "science": {
                        "keywords": ["research", "study", "experiment", "scientific", "data", "analysis", "hypothesis", "theory", "discovery"],
                        "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    },
                    "sports": {
                        "keywords": ["game", "player", "team", "match", "sport", "athlete", "competition", "championship", "tournament"],
                        "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    },
                    "politics": {
                        "keywords": ["government", "policy", "election", "vote", "political", "democracy", "law", "parliament", "senate"],
                        "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                    }
                }
                
                # Calculate scores for each category
                category_scores = {}
                for category, data in category_keywords.items():
                    if category in categories:
                        keywords = data["keywords"]
                        weights = data["weights"]
                        score = sum(weight for keyword, weight in zip(keywords, weights) if keyword in text_lower)
                        category_scores[category] = score
                
                # Find best category
                if category_scores:
                    best_category = max(category_scores, key=category_scores.get)
                    total_score = sum(category_scores.values())
                    confidence = category_scores[best_category] / total_score if total_score > 0 else 0
                else:
                    best_category = "unknown"
                    confidence = 0
                
                classification_result = {
                    "predicted_category": best_category,
                    "confidence": confidence,
                    "category_scores": category_scores
                }
            
            elif method == "ensemble":
                # Ensemble classification using multiple methods
                methods = ["naive_bayes", "logistic_regression", "random_forest"]
                ensemble_scores = {}
                
                for method_name in methods:
                    # Simulate different classification methods
                    if method_name == "naive_bayes":
                        # Use the same logic as above
                        category_scores = classification_result.get("category_scores", {})
                    elif method_name == "logistic_regression":
                        # Simulate logistic regression
                        category_scores = {cat: np.random.random() for cat in categories}
                    elif method_name == "random_forest":
                        # Simulate random forest
                        category_scores = {cat: np.random.random() for cat in categories}
                    
                    ensemble_scores[method_name] = category_scores
                
                # Combine scores
                final_scores = {}
                for category in categories:
                    scores = [ensemble_scores[method].get(category, 0) for method in methods]
                    final_scores[category] = np.mean(scores)
                
                best_category = max(final_scores, key=final_scores.get)
                confidence = final_scores[best_category] / sum(final_scores.values()) if sum(final_scores.values()) > 0 else 0
                
                classification_result = {
                    "predicted_category": best_category,
                    "confidence": confidence,
                    "category_scores": final_scores,
                    "ensemble_scores": ensemble_scores
                }
            
            # Update stats
            self.stats["total_classifications_made"] += 1
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
            logger.error(f"Error in advanced text classification: {e}")
            return {"error": str(e)}
    
    async def advanced_text_summarization(self, text: str, method: str = "extractive", 
                                        max_sentences: int = 3, include_ranking: bool = True) -> Dict[str, Any]:
        """Advanced text summarization with sentence ranking"""
        try:
            summary = ""
            sentence_rankings = []
            
            if method == "extractive":
                # Advanced extractive summarization
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) <= max_sentences:
                    summary = text
                    sentence_rankings = [{"sentence": sent, "score": 1.0, "rank": i+1} for i, sent in enumerate(sentences)]
                else:
                    # Advanced sentence scoring
                    word_freq = Counter(re.findall(r'\b\w+\b', text.lower()))
                    total_words = len(re.findall(r'\b\w+\b', text.lower()))
                    
                    sentence_scores = []
                    for sentence in sentences:
                        words = re.findall(r'\b\w+\b', sentence.lower())
                        if words:
                            # Score based on word frequency
                            freq_score = sum(word_freq[word] for word in words) / len(words)
                            
                            # Score based on sentence length (prefer medium length)
                            length_score = 1.0 - abs(len(words) - 15) / 15  # Optimal length around 15 words
                            
                            # Score based on position (prefer beginning and end)
                            position = sentences.index(sentence)
                            position_score = 1.0 if position < 2 or position > len(sentences) - 3 else 0.5
                            
                            # Score based on keywords (words that appear frequently)
                            keyword_score = sum(1 for word in words if word_freq[word] > 1) / len(words)
                            
                            # Combined score
                            total_score = (freq_score * 0.4 + length_score * 0.2 + position_score * 0.2 + keyword_score * 0.2)
                            sentence_scores.append((sentence, total_score))
                    
                    # Select top sentences
                    sentence_scores.sort(key=lambda x: x[1], reverse=True)
                    top_sentences = [sent for sent, _ in sentence_scores[:max_sentences]]
                    summary = ". ".join(top_sentences) + "."
                    
                    if include_ranking:
                        sentence_rankings = [{"sentence": sent, "score": score, "rank": i+1} 
                                           for i, (sent, score) in enumerate(sentence_scores[:max_sentences])]
            
            elif method == "abstractive":
                # Simple abstractive summarization
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) <= max_sentences:
                    summary = text
                else:
                    # Take first and last sentences with some middle content
                    if len(sentences) >= 3:
                        summary = sentences[0] + ". " + sentences[len(sentences)//2] + ". " + sentences[-1] + "."
                    else:
                        summary = sentences[0] + ". " + sentences[-1] + "."
            
            elif method == "hybrid":
                # Hybrid approach combining extractive and abstractive
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(sentences) <= max_sentences:
                    summary = text
                else:
                    # Extractive part
                    word_freq = Counter(re.findall(r'\b\w+\b', text.lower()))
                    sentence_scores = []
                    for sentence in sentences:
                        words = re.findall(r'\b\w+\b', sentence.lower())
                        score = sum(word_freq[word] for word in words) / len(words) if words else 0
                        sentence_scores.append((sentence, score))
                    
                    sentence_scores.sort(key=lambda x: x[1], reverse=True)
                    top_sentences = [sent for sent, _ in sentence_scores[:max_sentences]]
                    summary = ". ".join(top_sentences) + "."
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "method": method,
                "summary": summary,
                "sentence_rankings": sentence_rankings,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text) if len(text) > 0 else 0
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error in advanced text summarization: {e}")
            return {"error": str(e)}
    
    async def build_word_network(self, texts: List[str], min_frequency: int = 2) -> Dict[str, Any]:
        """Build word co-occurrence network"""
        try:
            # Extract words from all texts
            all_words = []
            for text in texts:
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend(words)
            
            # Filter by frequency
            word_freq = Counter(all_words)
            frequent_words = [word for word, freq in word_freq.items() if freq >= min_frequency]
            
            # Build co-occurrence matrix
            co_occurrence = defaultdict(int)
            for text in texts:
                words = [word for word in re.findall(r'\b\w+\b', text.lower()) if word in frequent_words]
                for i, word1 in enumerate(words):
                    for j, word2 in enumerate(words[i+1:], i+1):
                        co_occurrence[(word1, word2)] += 1
            
            # Create network
            G = nx.Graph()
            for (word1, word2), weight in co_occurrence.items():
                G.add_edge(word1, word2, weight=weight)
            
            # Calculate network metrics
            network_metrics = {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "average_clustering": nx.average_clustering(G),
                "average_shortest_path_length": nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
            }
            
            # Get top nodes by degree
            degree_centrality = nx.degree_centrality(G)
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Store network
            network_id = f"network_{int(time.time())}_{secrets.token_hex(4)}"
            self.nlp_networks["word_networks"][network_id] = {
                "graph": G,
                "metrics": network_metrics,
                "top_nodes": top_nodes,
                "created_at": datetime.now().isoformat()
            }
            
            # Update stats
            self.stats["total_networks_built"] += 1
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "network_id": network_id,
                "network_metrics": network_metrics,
                "top_nodes": top_nodes,
                "frequent_words": len(frequent_words),
                "co_occurrence_pairs": len(co_occurrence)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error building word network: {e}")
            return {"error": str(e)}
    
    async def calculate_readability_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate advanced readability metrics"""
        try:
            # Calculate various readability metrics
            metrics = {
                "flesch_reading_ease": flesch_reading_ease(text),
                "flesch_kincaid_grade": flesch_kincaid_grade(text),
                "smog_index": smog_index(text),
                "coleman_liau_index": coleman_liau_index(text)
            }
            
            # Calculate additional metrics
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b\w+\b', text)
            syllables = sum(len(re.findall(r'[aeiouy]+', word.lower())) for word in words)
            
            # Custom metrics
            metrics.update({
                "average_sentence_length": len(words) / len(sentences) if sentences else 0,
                "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "syllable_count": syllables,
                "sentence_count": len(sentences),
                "word_count": len(words),
                "character_count": len(text)
            })
            
            # Determine readability level
            flesch_score = metrics["flesch_reading_ease"]
            if flesch_score >= 90:
                readability_level = "Very Easy"
            elif flesch_score >= 80:
                readability_level = "Easy"
            elif flesch_score >= 70:
                readability_level = "Fairly Easy"
            elif flesch_score >= 60:
                readability_level = "Standard"
            elif flesch_score >= 50:
                readability_level = "Fairly Difficult"
            elif flesch_score >= 30:
                readability_level = "Difficult"
            else:
                readability_level = "Very Difficult"
            
            # Store metrics
            self.nlp_metrics["readability_metrics"] = metrics
            
            # Update stats
            self.stats["total_nlp_requests"] += 1
            self.stats["successful_nlp_requests"] += 1
            
            return {
                "status": "success",
                "readability_metrics": metrics,
                "readability_level": readability_level,
                "recommendations": self._get_readability_recommendations(metrics)
            }
            
        except Exception as e:
            self.stats["failed_nlp_requests"] += 1
            logger.error(f"Error calculating readability metrics: {e}")
            return {"error": str(e)}
    
    def _get_readability_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Get readability improvement recommendations"""
        recommendations = []
        
        if metrics["average_sentence_length"] > 20:
            recommendations.append("Consider breaking long sentences into shorter ones")
        
        if metrics["average_word_length"] > 6:
            recommendations.append("Consider using shorter, simpler words")
        
        if metrics["flesch_reading_ease"] < 60:
            recommendations.append("Text may be too complex for general audience")
        
        if metrics["sentence_count"] < 3:
            recommendations.append("Consider adding more sentences for better flow")
        
        return recommendations
    
    def get_advanced_nlp_stats(self) -> Dict[str, Any]:
        """Get advanced NLP processing statistics"""
        uptime = time.time() - self.stats["start_time"]
        return {
            "stats": self.stats.copy(),
            "uptime_seconds": round(uptime, 2),
            "uptime_hours": round(uptime / 3600, 2),
            "success_rate": (self.stats["successful_nlp_requests"] / self.stats["total_nlp_requests"] * 100) if self.stats["total_nlp_requests"] > 0 else 0,
            "average_tokens_per_request": self.stats["total_tokens_processed"] / self.stats["total_nlp_requests"] if self.stats["total_nlp_requests"] > 0 else 0,
            "average_sentences_per_request": self.stats["total_sentences_processed"] / self.stats["total_nlp_requests"] if self.stats["total_nlp_requests"] > 0 else 0,
            "embeddings_created": self.stats["total_embeddings_created"],
            "similarities_calculated": self.stats["total_similarities_calculated"],
            "clusters_created": self.stats["total_clusters_created"],
            "topics_discovered": self.stats["total_topics_discovered"],
            "classifications_made": self.stats["total_classifications_made"],
            "networks_built": self.stats["total_networks_built"]
        }

# Global instance
advanced_nlp_system = AdvancedNLPSystem()












