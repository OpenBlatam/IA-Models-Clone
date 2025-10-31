"""
Advanced Machine Learning Pipeline Service for content analysis and optimization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

from ..models.database import BlogPost, User, Comment, Like, View
from ..models.schemas import PostStatus
from ..core.exceptions import DatabaseError, ValidationError


class MLPipelineService:
    """Service for advanced machine learning operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.models = {}
        self.vectorizers = {}
        self.scalers = {}
        
        # Initialize ML components
        self._initialize_ml_components()
    
    def _initialize_ml_components(self):
        """Initialize machine learning components."""
        try:
            # Text vectorizers
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.vectorizers['count'] = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Clustering models
            self.models['kmeans'] = KMeans(n_clusters=10, random_state=42)
            self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
            
            # Topic modeling
            self.models['lda'] = LatentDirichletAllocation(
                n_components=10,
                random_state=42
            )
            
            self.models['nmf'] = NMF(
                n_components=10,
                random_state=42
            )
            
            # Classification and regression
            self.models['content_classifier'] = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            self.models['engagement_predictor'] = GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
            
            # Scalers
            self.scalers['standard'] = StandardScaler()
            
        except Exception as e:
            print(f"Warning: Could not initialize some ML components: {e}")
    
    async def train_content_classifier(self) -> Dict[str, Any]:
        """Train a content classification model."""
        try:
            # Get training data
            posts_query = select(BlogPost).where(
                and_(
                    BlogPost.status == PostStatus.PUBLISHED.value,
                    BlogPost.category.isnot(None)
                )
            )
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.scalars().all()
            
            if len(posts) < 50:
                raise ValidationError("Insufficient training data")
            
            # Prepare features and labels
            texts = []
            labels = []
            
            for post in posts:
                texts.append(f"{post.title} {post.content}")
                labels.append(post.category)
            
            # Vectorize text
            X = self.vectorizers['tfidf'].fit_transform(texts)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.models['content_classifier'].fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.models['content_classifier'].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            await self._save_model('content_classifier', self.models['content_classifier'])
            
            return {
                "model_name": "content_classifier",
                "accuracy": accuracy,
                "training_samples": len(posts),
                "test_samples": len(X_test),
                "feature_count": X.shape[1],
                "categories": list(set(labels))
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to train content classifier: {str(e)}")
    
    async def train_engagement_predictor(self) -> Dict[str, Any]:
        """Train an engagement prediction model."""
        try:
            # Get training data
            posts_query = select(BlogPost).where(
                BlogPost.status == PostStatus.PUBLISHED.value
            )
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.scalars().all()
            
            if len(posts) < 50:
                raise ValidationError("Insufficient training data")
            
            # Prepare features
            features = []
            targets = []
            
            for post in posts:
                # Text features
                text = f"{post.title} {post.content}"
                text_vector = self.vectorizers['tfidf'].fit_transform([text])
                
                # Numerical features
                numerical_features = [
                    post.word_count,
                    post.reading_time_minutes,
                    len(post.tags) if post.tags else 0,
                    post.view_count,
                    post.like_count,
                    post.comment_count
                ]
                
                # Combine features
                feature_vector = np.concatenate([
                    text_vector.toarray().flatten(),
                    numerical_features
                ])
                
                features.append(feature_vector)
                
                # Engagement score (weighted combination)
                engagement_score = (
                    post.view_count * 0.1 +
                    post.like_count * 0.3 +
                    post.comment_count * 0.6
                )
                targets.append(engagement_score)
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scalers['standard'].fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.models['engagement_predictor'].fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.models['engagement_predictor'].predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Save model
            await self._save_model('engagement_predictor', self.models['engagement_predictor'])
            
            return {
                "model_name": "engagement_predictor",
                "rmse": rmse,
                "mse": mse,
                "training_samples": len(posts),
                "test_samples": len(X_test),
                "feature_count": X.shape[1]
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to train engagement predictor: {str(e)}")
    
    async def perform_topic_modeling(self, num_topics: int = 10) -> Dict[str, Any]:
        """Perform topic modeling on blog posts."""
        try:
            # Get posts
            posts_query = select(BlogPost).where(
                BlogPost.status == PostStatus.PUBLISHED.value
            )
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.scalars().all()
            
            if len(posts) < 20:
                raise ValidationError("Insufficient data for topic modeling")
            
            # Prepare texts
            texts = []
            for post in posts:
                texts.append(f"{post.title} {post.content}")
            
            # Vectorize
            tfidf_matrix = self.vectorizers['tfidf'].fit_transform(texts)
            
            # LDA topic modeling
            self.models['lda'].n_components = num_topics
            lda_topics = self.models['lda'].fit_transform(tfidf_matrix)
            
            # NMF topic modeling
            self.models['nmf'].n_components = num_topics
            nmf_topics = self.models['nmf'].fit_transform(tfidf_matrix)
            
            # Extract topic words
            feature_names = self.vectorizers['tfidf'].get_feature_names_out()
            
            lda_topic_words = []
            for topic_idx, topic in enumerate(self.models['lda'].components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                lda_topic_words.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            nmf_topic_words = []
            for topic_idx, topic in enumerate(self.models['nmf'].components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                nmf_topic_words.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": topic[top_words_idx].tolist()
                })
            
            return {
                "num_topics": num_topics,
                "lda_topics": lda_topic_words,
                "nmf_topics": nmf_topic_words,
                "posts_analyzed": len(posts),
                "feature_count": len(feature_names)
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to perform topic modeling: {str(e)}")
    
    async def perform_content_clustering(self, num_clusters: int = 10) -> Dict[str, Any]:
        """Perform content clustering."""
        try:
            # Get posts
            posts_query = select(BlogPost).where(
                BlogPost.status == PostStatus.PUBLISHED.value
            )
            posts_result = await self.session.execute(posts_query)
            posts = posts_result.scalars().all()
            
            if len(posts) < 20:
                raise ValidationError("Insufficient data for clustering")
            
            # Prepare texts
            texts = []
            post_ids = []
            for post in posts:
                texts.append(f"{post.title} {post.content}")
                post_ids.append(post.id)
            
            # Vectorize
            tfidf_matrix = self.vectorizers['tfidf'].fit_transform(texts)
            
            # K-means clustering
            self.models['kmeans'].n_clusters = num_clusters
            kmeans_labels = self.models['kmeans'].fit_predict(tfidf_matrix)
            
            # DBSCAN clustering
            dbscan_labels = self.models['dbscan'].fit_predict(tfidf_matrix)
            
            # Organize clusters
            kmeans_clusters = {}
            for i, label in enumerate(kmeans_labels):
                if label not in kmeans_clusters:
                    kmeans_clusters[label] = []
                kmeans_clusters[label].append({
                    "post_id": post_ids[i],
                    "title": posts[i].title
                })
            
            dbscan_clusters = {}
            for i, label in enumerate(dbscan_labels):
                if label == -1:  # Noise points
                    continue
                if label not in dbscan_clusters:
                    dbscan_clusters[label] = []
                dbscan_clusters[label].append({
                    "post_id": post_ids[i],
                    "title": posts[i].title
                })
            
            return {
                "num_clusters": num_clusters,
                "kmeans_clusters": kmeans_clusters,
                "dbscan_clusters": dbscan_clusters,
                "posts_clustered": len(posts),
                "noise_points": list(dbscan_labels).count(-1)
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to perform content clustering: {str(e)}")
    
    async def predict_content_engagement(
        self,
        title: str,
        content: str,
        tags: List[str] = None
    ) -> Dict[str, Any]:
        """Predict content engagement."""
        try:
            if not self.models.get('engagement_predictor'):
                raise ValidationError("Engagement predictor not trained")
            
            # Prepare features
            text = f"{title} {content}"
            text_vector = self.vectorizers['tfidf'].transform([text])
            
            # Numerical features
            numerical_features = [
                len(content.split()),  # word count
                len(content.split()) // 200,  # reading time estimate
                len(tags) if tags else 0,  # tag count
                0,  # view count (unknown for new content)
                0,  # like count (unknown for new content)
                0   # comment count (unknown for new content)
            ]
            
            # Combine features
            feature_vector = np.concatenate([
                text_vector.toarray().flatten(),
                numerical_features
            ]).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scalers['standard'].transform(feature_vector)
            
            # Predict
            engagement_score = self.models['engagement_predictor'].predict(feature_vector_scaled)[0]
            
            # Get feature importance
            feature_importance = self.models['engagement_predictor'].feature_importances_
            
            return {
                "predicted_engagement": float(engagement_score),
                "confidence": "medium",  # Could be calculated based on model uncertainty
                "feature_importance": feature_importance.tolist(),
                "model_used": "gradient_boosting_regressor"
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to predict content engagement: {str(e)}")
    
    async def classify_content(
        self,
        title: str,
        content: str
    ) -> Dict[str, Any]:
        """Classify content into categories."""
        try:
            if not self.models.get('content_classifier'):
                raise ValidationError("Content classifier not trained")
            
            # Prepare text
            text = f"{title} {content}"
            text_vector = self.vectorizers['tfidf'].transform([text])
            
            # Predict
            prediction = self.models['content_classifier'].predict(text_vector)[0]
            probabilities = self.models['content_classifier'].predict_proba(text_vector)[0]
            
            # Get class names
            class_names = self.models['content_classifier'].classes_
            
            # Create probability mapping
            prob_mapping = {}
            for i, class_name in enumerate(class_names):
                prob_mapping[class_name] = float(probabilities[i])
            
            return {
                "predicted_category": prediction,
                "confidence": float(max(probabilities)),
                "all_probabilities": prob_mapping,
                "model_used": "random_forest_classifier"
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to classify content: {str(e)}")
    
    async def get_ml_pipeline_stats(self) -> Dict[str, Any]:
        """Get ML pipeline statistics."""
        try:
            # Get model status
            model_status = {}
            for model_name, model in self.models.items():
                model_status[model_name] = {
                    "trained": hasattr(model, 'feature_importances_') or hasattr(model, 'components_'),
                    "type": type(model).__name__
                }
            
            # Get data statistics
            posts_count_query = select(func.count(BlogPost.id)).where(
                BlogPost.status == PostStatus.PUBLISHED.value
            )
            posts_count_result = await self.session.execute(posts_count_query)
            posts_count = posts_count_result.scalar()
            
            users_count_query = select(func.count(User.id))
            users_count_result = await self.session.execute(users_count_query)
            users_count = users_count_result.scalar()
            
            comments_count_query = select(func.count(Comment.id))
            comments_count_result = await self.session.execute(comments_count_query)
            comments_count = comments_count_result.scalar()
            
            return {
                "model_status": model_status,
                "data_statistics": {
                    "total_posts": posts_count,
                    "total_users": users_count,
                    "total_comments": comments_count
                },
                "available_models": list(self.models.keys()),
                "available_vectorizers": list(self.vectorizers.keys()),
                "available_scalers": list(self.scalers.keys())
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get ML pipeline stats: {str(e)}")
    
    async def _save_model(self, model_name: str, model: Any):
        """Save a trained model."""
        try:
            # In a real implementation, you would save to a model store
            # For now, we'll just store in memory
            self.models[model_name] = model
            
            # You could also save to disk:
            # joblib.dump(model, f"models/{model_name}.joblib")
            
        except Exception as e:
            print(f"Warning: Could not save model {model_name}: {e}")
    
    async def load_model(self, model_name: str) -> bool:
        """Load a trained model."""
        try:
            # In a real implementation, you would load from a model store
            # For now, we'll just check if it exists in memory
            return model_name in self.models
            
        except Exception as e:
            print(f"Warning: Could not load model {model_name}: {e}")
            return False
    
    async def retrain_models(self) -> Dict[str, Any]:
        """Retrain all models with latest data."""
        try:
            results = {}
            
            # Retrain content classifier
            try:
                classifier_result = await self.train_content_classifier()
                results['content_classifier'] = classifier_result
            except Exception as e:
                results['content_classifier'] = {"error": str(e)}
            
            # Retrain engagement predictor
            try:
                predictor_result = await self.train_engagement_predictor()
                results['engagement_predictor'] = predictor_result
            except Exception as e:
                results['engagement_predictor'] = {"error": str(e)}
            
            return {
                "retraining_completed": True,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to retrain models: {str(e)}")






























