"""
Realistic AI/ML Libraries - Librerías de IA/ML que realmente existen
================================================================

Librerías de IA y ML que realmente existen, están actualizadas
y funcionan en la práctica.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LibraryInfo:
    """Información real de una librería."""
    name: str
    version: str
    description: str
    use_case: str
    pros: List[str]
    cons: List[str]
    installation: str
    example: str
    real_usage: str
    alternatives: List[str]
    documentation: str
    github: str
    pypi: str


class RealisticAIMLLibraries:
    """
    Librerías de IA/ML realistas que realmente existen.
    """
    
    def __init__(self):
        """Inicializar con librerías de IA/ML reales."""
        self.libraries = {
            'scikit-learn': LibraryInfo(
                name="scikit-learn",
                version="1.3.2",
                description="Machine Learning para Python",
                use_case="Clasificación, regresión, clustering, análisis de texto",
                pros=[
                    "API consistente y fácil de usar",
                    "Amplia gama de algoritmos ML",
                    "Excelente documentación",
                    "Optimizado para rendimiento",
                    "Integración con NumPy y pandas"
                ],
                cons=[
                    "Limitado para deep learning",
                    "Puede ser lento para datasets muy grandes"
                ],
                installation="pip install scikit-learn",
                example="""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Datos de ejemplo
texts = [
    "This is a great product",
    "I love this amazing product",
    "This product is terrible",
    "I hate this awful product"
]
labels = [1, 1, 0, 0]  # 1: positivo, 0: negativo

# Vectorización
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Similitud
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Similarity: {similarity[0][0]:.3f}")

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
print(f"Clusters: {clusters}")

# Clasificación
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, labels, test_size=0.5, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
""",
                real_usage="""
# Uso real en producción
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import joblib

class TextAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clusterer = KMeans(n_clusters=5, random_state=42)
        self.is_fitted = False
    
    def fit(self, texts):
        """Entrenar el modelo"""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.clusterer.fit(tfidf_matrix)
        self.is_fitted = True
    
    def predict_similarity(self, text1, text2):
        """Calcular similitud entre dos textos"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        texts = [text1, text2]
        tfidf_matrix = self.vectorizer.transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]
    
    def cluster_texts(self, texts):
        """Agrupar textos en clusters"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        tfidf_matrix = self.vectorizer.transform(texts)
        clusters = self.clusterer.predict(tfidf_matrix)
        return clusters
    
    def save_model(self, filepath):
        """Guardar modelo"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'clusterer': self.clusterer,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load_model(self, filepath):
        """Cargar modelo"""
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.clusterer = model_data['clusterer']
        self.is_fitted = model_data['is_fitted']

# Uso en aplicación
analyzer = TextAnalyzer()
texts = ["text1", "text2", "text3"]  # Datos de entrenamiento
analyzer.fit(texts)

# Calcular similitud
similarity = analyzer.predict_similarity("text1", "text2")
print(f"Similarity: {similarity:.3f}")

# Guardar modelo
analyzer.save_model('text_analyzer.pkl')
""",
                alternatives=["xgboost", "lightgbm", "catboost"],
                documentation="https://scikit-learn.org/stable/",
                github="https://github.com/scikit-learn/scikit-learn",
                pypi="https://pypi.org/project/scikit-learn/"
            ),
            
            'transformers': LibraryInfo(
                name="transformers",
                version="4.35.2",
                description="Modelos de transformers de Hugging Face",
                use_case="Modelos de lenguaje, análisis de sentimientos, generación de texto",
                pros=[
                    "Acceso a miles de modelos pre-entrenados",
                    "API simple y consistente",
                    "Soporte para múltiples frameworks",
                    "Modelos state-of-the-art"
                ],
                cons=[
                    "Modelos pueden ser muy grandes",
                    "Requiere GPU para mejor rendimiento",
                    "Uso de memoria alto"
                ],
                installation="pip install transformers torch",
                example="""
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Análisis de sentimientos
sentiment_analyzer = pipeline("sentiment-analysis")
text = "I love this product, it's amazing!"
result = sentiment_analyzer(text)
print(f"Sentiment: {result}")

# Generación de texto
text_generator = pipeline("text-generation", model="gpt2")
generated = text_generator("The future of AI is", max_length=50, num_return_sequences=1)
print(f"Generated: {generated[0]['generated_text']}")

# Extracción de entidades
ner_pipeline = pipeline("ner", aggregation_strategy="simple")
entities = ner_pipeline("Apple is looking at buying U.K. startup for $1 billion")
print(f"Entities: {entities}")

# Embeddings
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

embeddings = get_embeddings("This is a sample text")
print(f"Embedding shape: {embeddings.shape}")
""",
                real_usage="""
# Uso real en producción
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class TextSimilarityAnalyzer:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def get_embeddings(self, texts):
        """Obtener embeddings para una lista de textos"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(embedding[0])
        return np.array(embeddings)
    
    def calculate_similarity(self, text1, text2):
        """Calcular similitud entre dos textos"""
        embeddings = self.get_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def analyze_sentiment(self, text):
        """Analizar sentimiento de un texto"""
        result = self.sentiment_analyzer(text)
        return {
            'label': result[0]['label'],
            'score': result[0]['score']
        }
    
    def find_similar_texts(self, query_text, texts, threshold=0.7):
        """Encontrar textos similares a una consulta"""
        query_embedding = self.get_embeddings([query_text])[0]
        text_embeddings = self.get_embeddings(texts)
        
        similarities = cosine_similarity([query_embedding], text_embeddings)[0]
        similar_indices = np.where(similarities >= threshold)[0]
        
        results = []
        for idx in similar_indices:
            results.append({
                'text': texts[idx],
                'similarity': similarities[idx]
            })
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Uso en aplicación
analyzer = TextSimilarityAnalyzer()

# Calcular similitud
similarity = analyzer.calculate_similarity(
    "I love this product",
    "This product is amazing"
)
print(f"Similarity: {similarity:.3f}")

# Analizar sentimiento
sentiment = analyzer.analyze_sentiment("I love this product!")
print(f"Sentiment: {sentiment}")

# Encontrar textos similares
texts = ["text1", "text2", "text3"]
similar = analyzer.find_similar_texts("query text", texts)
print(f"Similar texts: {similar}")
""",
                alternatives=["openai", "anthropic", "cohere"],
                documentation="https://huggingface.co/docs/transformers/",
                github="https://github.com/huggingface/transformers",
                pypi="https://pypi.org/project/transformers/"
            ),
            
            'sentence-transformers': LibraryInfo(
                name="sentence-transformers",
                version="2.2.2",
                description="Embeddings de oraciones optimizados",
                use_case="Similitud de texto, búsqueda semántica, clustering",
                pros=[
                    "Modelos optimizados para embeddings",
                    "Fácil de usar",
                    "Buena calidad de embeddings",
                    "Soporte para múltiples idiomas"
                ],
                cons=[
                    "Modelos pueden ser grandes",
                    "Requiere GPU para mejor rendimiento"
                ],
                installation="pip install sentence-transformers",
                example="""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Cargar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generar embeddings
sentences = [
    "This is a test sentence",
    "This is another test sentence",
    "I love this product",
    "This product is amazing"
]

embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")

# Calcular similitud
similarity_matrix = cosine_similarity(embeddings)
print("Similarity matrix:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i != j:
            similarity = similarity_matrix[i][j]
            print(f"'{sent1}' <-> '{sent2}': {similarity:.3f}")

# Búsqueda semántica
query = "I love this amazing product"
query_embedding = model.encode([query])
query_similarity = cosine_similarity(query_embedding, embeddings)[0]

most_similar_idx = np.argmax(query_similarity)
print(f"\\nQuery: '{query}'")
print(f"Most similar: '{sentences[most_similar_idx]}' (similarity: {query_similarity[most_similar_idx]:.3f})")
""",
                real_usage="""
# Uso real en producción
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import joblib

class SemanticTextAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.clusterer = None
        self.is_fitted = False
    
    def encode_texts(self, texts):
        """Codificar textos a embeddings"""
        return self.model.encode(texts)
    
    def calculate_similarity(self, text1, text2):
        """Calcular similitud entre dos textos"""
        embeddings = self.encode_texts([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def find_similar_texts(self, query_text, texts, top_k=5):
        """Encontrar textos más similares a una consulta"""
        query_embedding = self.encode_texts([query_text])
        text_embeddings = self.encode_texts(texts)
        
        similarities = cosine_similarity(query_embedding, text_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': texts[idx],
                'similarity': similarities[idx]
            })
        
        return results
    
    def cluster_texts(self, texts, n_clusters=5):
        """Agrupar textos en clusters semánticos"""
        embeddings = self.encode_texts(texts)
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.clusterer.fit_predict(embeddings)
        self.is_fitted = True
        
        # Crear DataFrame con resultados
        df = pd.DataFrame({
            'text': texts,
            'cluster': clusters
        })
        
        return df
    
    def predict_cluster(self, text):
        """Predecir cluster para un nuevo texto"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        embedding = self.encode_texts([text])
        cluster = self.clusterer.predict(embedding)[0]
        return cluster
    
    def save_model(self, filepath):
        """Guardar modelo"""
        joblib.dump({
            'clusterer': self.clusterer,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load_model(self, filepath):
        """Cargar modelo"""
        model_data = joblib.load(filepath)
        self.clusterer = model_data['clusterer']
        self.is_fitted = model_data['is_fitted']

# Uso en aplicación
analyzer = SemanticTextAnalyzer()

# Calcular similitud
similarity = analyzer.calculate_similarity(
    "I love this product",
    "This product is amazing"
)
print(f"Similarity: {similarity:.3f}")

# Encontrar textos similares
texts = ["text1", "text2", "text3", "text4", "text5"]
similar = analyzer.find_similar_texts("query text", texts, top_k=3)
print(f"Similar texts: {similar}")

# Clustering
clustered_df = analyzer.cluster_texts(texts, n_clusters=3)
print(clustered_df)

# Guardar modelo
analyzer.save_model('semantic_analyzer.pkl')
""",
                alternatives=["transformers", "openai-embeddings", "cohere-embeddings"],
                documentation="https://www.sbert.net/",
                github="https://github.com/UKPLab/sentence-transformers",
                pypi="https://pypi.org/project/sentence-transformers/"
            ),
            
            'textblob': LibraryInfo(
                name="textblob",
                version="0.17.1",
                description="Procesamiento de texto simple",
                use_case="Análisis de sentimientos, corrección ortográfica, análisis de texto",
                pros=[
                    "Muy fácil de usar",
                    "API simple",
                    "Bueno para prototipado",
                    "Incluye corrección ortográfica"
                ],
                cons=[
                    "Limitado en funcionalidades",
                    "No tan preciso como otras librerías"
                ],
                installation="pip install textblob",
                example="""
from textblob import TextBlob

# Análisis de sentimientos
text = TextBlob("I love this product!")
sentiment = text.sentiment
print(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

# Corrección ortográfica
text = TextBlob("I lov this prodct!")
corrected = text.correct()
print(f"Corrected: {corrected}")

# Análisis de texto
text = TextBlob("This is a sample text for analysis.")
print(f"Words: {text.words}")
print(f"Sentences: {text.sentences}")
print(f"Noun phrases: {text.noun_phrases}")

# Traducción
text = TextBlob("Hello, how are you?")
translated = text.translate(to='es')
print(f"Translated: {translated}")
""",
                real_usage="""
# Uso real en aplicación
from textblob import TextBlob
import pandas as pd

class TextBlobAnalyzer:
    def __init__(self):
        pass
    
    def analyze_sentiment(self, text):
        """Analizar sentimiento de un texto"""
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        return {
            'polarity': sentiment.polarity,
            'subjectivity': sentiment.subjectivity,
            'sentiment_label': self._get_sentiment_label(sentiment.polarity)
        }
    
    def _get_sentiment_label(self, polarity):
        """Convertir polaridad a etiqueta"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def correct_text(self, text):
        """Corregir ortografía de un texto"""
        blob = TextBlob(text)
        corrected = blob.correct()
        return str(corrected)
    
    def extract_noun_phrases(self, text):
        """Extraer frases nominales"""
        blob = TextBlob(text)
        return list(blob.noun_phrases)
    
    def analyze_texts_batch(self, texts):
        """Analizar múltiples textos"""
        results = []
        for text in texts:
            analysis = self.analyze_sentiment(text)
            analysis['text'] = text
            results.append(analysis)
        
        return pd.DataFrame(results)

# Uso en aplicación
analyzer = TextBlobAnalyzer()

# Analizar sentimiento
sentiment = analyzer.analyze_sentiment("I love this product!")
print(f"Sentiment: {sentiment}")

# Corregir texto
corrected = analyzer.correct_text("I lov this prodct!")
print(f"Corrected: {corrected}")

# Extraer frases nominales
phrases = analyzer.extract_noun_phrases("The quick brown fox jumps over the lazy dog")
print(f"Noun phrases: {phrases}")

# Análisis en lote
texts = ["I love this!", "This is terrible", "It's okay"]
results_df = analyzer.analyze_texts_batch(texts)
print(results_df)
""",
                alternatives=["vaderSentiment", "nltk", "spacy"],
                documentation="https://textblob.readthedocs.io/",
                github="https://github.com/sloria/TextBlob",
                pypi="https://pypi.org/project/textblob/"
            ),
            
            'vaderSentiment': LibraryInfo(
                name="vaderSentiment",
                version="3.3.2",
                description="Análisis de sentimientos para redes sociales",
                use_case="Análisis de sentimientos, texto informal, redes sociales",
                pros=[
                    "Optimizado para texto informal",
                    "Maneja emojis y slang",
                    "No requiere entrenamiento",
                    "Rápido y eficiente"
                ],
                cons=[
                    "Específico para inglés",
                    "Limitado a análisis de sentimientos"
                ],
                installation="pip install vaderSentiment",
                example="""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Inicializar analizador
analyzer = SentimentIntensityAnalyzer()

# Analizar sentimientos
texts = [
    "I love this product! 😍",
    "This is terrible :(",
    "It's okay, nothing special",
    "AMAZING!!! 🔥🔥🔥",
    "Meh, it's fine I guess"
]

for text in texts:
    scores = analyzer.polarity_scores(text)
    print(f"Text: '{text}'")
    print(f"Scores: {scores}")
    print(f"Sentiment: {scores['compound']:.3f}")
    print()

# Función para interpretar sentimiento
def get_sentiment_label(compound_score):
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Ejemplo de uso
text = "I love this product! 😍"
scores = analyzer.polarity_scores(text)
sentiment = get_sentiment_label(scores['compound'])
print(f"Text: '{text}'")
print(f"Sentiment: {sentiment} (score: {scores['compound']:.3f})")
""",
                real_usage="""
# Uso real en aplicación
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text):
        """Analizar sentimiento de un texto"""
        scores = self.analyzer.polarity_scores(text)
        
        return {
            'text': text,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound'],
            'sentiment_label': self._get_sentiment_label(scores['compound'])
        }
    
    def _get_sentiment_label(self, compound_score):
        """Convertir score compuesto a etiqueta"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_texts_batch(self, texts):
        """Analizar múltiples textos"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_sentiment_distribution(self, texts):
        """Obtener distribución de sentimientos"""
        df = self.analyze_texts_batch(texts)
        distribution = df['sentiment_label'].value_counts()
        return distribution
    
    def filter_by_sentiment(self, texts, sentiment_label):
        """Filtrar textos por sentimiento"""
        df = self.analyze_texts_batch(texts)
        filtered = df[df['sentiment_label'] == sentiment_label]
        return filtered['text'].tolist()

# Uso en aplicación
analyzer = VaderSentimentAnalyzer()

# Analizar sentimiento
sentiment = analyzer.analyze_sentiment("I love this product! 😍")
print(f"Sentiment: {sentiment}")

# Análisis en lote
texts = [
    "I love this product! 😍",
    "This is terrible :(",
    "It's okay, nothing special",
    "AMAZING!!! 🔥🔥🔥",
    "Meh, it's fine I guess"
]

results_df = analyzer.analyze_texts_batch(texts)
print(results_df)

# Distribución de sentimientos
distribution = analyzer.get_sentiment_distribution(texts)
print(f"\\nSentiment distribution:")
print(distribution)

# Filtrar por sentimiento
positive_texts = analyzer.filter_by_sentiment(texts, 'positive')
print(f"\\nPositive texts: {positive_texts}")
""",
                alternatives=["textblob", "nltk", "transformers"],
                documentation="https://github.com/cjhutto/vaderSentiment",
                github="https://github.com/cjhutto/vaderSentiment",
                pypi="https://pypi.org/project/vaderSentiment/"
            ),
            
            'textstat': LibraryInfo(
                name="textstat",
                version="0.7.3",
                description="Métricas de legibilidad y complejidad de texto",
                use_case="Análisis de legibilidad, métricas de texto, evaluación de calidad",
                pros=[
                    "Múltiples métricas de legibilidad",
                    "Fácil de usar",
                    "Soporte para múltiples idiomas",
                    "Métricas estándar de la industria"
                ],
                cons=[
                    "Limitado a métricas de legibilidad",
                    "Puede ser lento para textos muy largos"
                ],
                installation="pip install textstat",
                example="""
import textstat

text = "This is a sample text for readability analysis. It contains multiple sentences to demonstrate various readability metrics."

# Métricas de legibilidad
flesch_score = textstat.flesch_reading_ease(text)
flesch_kincaid = textstat.flesch_kincaid_grade(text)
gunning_fog = textstat.gunning_fog(text)
smog_index = textstat.smog_index(text)
coleman_liau = textstat.coleman_liau_index(text)
automated_readability = textstat.automated_readability_index(text)

print(f"Flesch Reading Ease: {flesch_score}")
print(f"Flesch-Kincaid Grade: {flesch_kincaid}")
print(f"Gunning Fog: {gunning_fog}")
print(f"SMOG Index: {smog_index}")
print(f"Coleman-Liau Index: {coleman_liau}")
print(f"Automated Readability Index: {automated_readability}")

# Métricas de complejidad
syllable_count = textstat.syllable_count(text)
lexicon_count = textstat.lexicon_count(text)
sentence_count = textstat.sentence_count(text)
char_count = textstat.char_count(text)
letter_count = textstat.letter_count(text)
polysyllable_count = textstat.polysyllable_count(text)

print(f"\\nSyllable count: {syllable_count}")
print(f"Lexicon count: {lexicon_count}")
print(f"Sentence count: {sentence_count}")
print(f"Character count: {char_count}")
print(f"Letter count: {letter_count}")
print(f"Polysyllable count: {polysyllable_count}")
""",
                real_usage="""
# Uso real en aplicación
import textstat
import pandas as pd
import numpy as np

class TextReadabilityAnalyzer:
    def __init__(self):
        self.metrics = [
            'flesch_reading_ease',
            'flesch_kincaid_grade',
            'gunning_fog',
            'smog_index',
            'coleman_liau_index',
            'automated_readability_index'
        ]
    
    def analyze_readability(self, text):
        """Analizar legibilidad de un texto"""
        results = {
            'text': text,
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'syllable_count': textstat.syllable_count(text),
            'lexicon_count': textstat.lexicon_count(text),
            'sentence_count': textstat.sentence_count(text),
            'char_count': textstat.char_count(text),
            'letter_count': textstat.letter_count(text),
            'polysyllable_count': textstat.polysyllable_count(text)
        }
        
        # Agregar interpretación de legibilidad
        results['readability_level'] = self._interpret_readability(results['flesch_reading_ease'])
        
        return results
    
    def _interpret_readability(self, flesch_score):
        """Interpretar score de Flesch Reading Ease"""
        if flesch_score >= 90:
            return 'Very Easy'
        elif flesch_score >= 80:
            return 'Easy'
        elif flesch_score >= 70:
            return 'Fairly Easy'
        elif flesch_score >= 60:
            return 'Standard'
        elif flesch_score >= 50:
            return 'Fairly Difficult'
        elif flesch_score >= 30:
            return 'Difficult'
        else:
            return 'Very Difficult'
    
    def analyze_texts_batch(self, texts):
        """Analizar múltiples textos"""
        results = []
        for text in texts:
            result = self.analyze_readability(text)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_readability_summary(self, texts):
        """Obtener resumen de legibilidad"""
        df = self.analyze_texts_batch(texts)
        
        summary = {
            'total_texts': len(texts),
            'avg_flesch_score': df['flesch_reading_ease'].mean(),
            'avg_grade_level': df['flesch_kincaid_grade'].mean(),
            'readability_distribution': df['readability_level'].value_counts().to_dict(),
            'avg_syllable_count': df['syllable_count'].mean(),
            'avg_lexicon_count': df['lexicon_count'].mean(),
            'avg_sentence_count': df['sentence_count'].mean()
        }
        
        return summary
    
    def compare_readability(self, text1, text2):
        """Comparar legibilidad de dos textos"""
        analysis1 = self.analyze_readability(text1)
        analysis2 = self.analyze_readability(text2)
        
        comparison = {
            'text1': {
                'flesch_score': analysis1['flesch_reading_ease'],
                'grade_level': analysis1['flesch_kincaid_grade'],
                'readability_level': analysis1['readability_level']
            },
            'text2': {
                'flesch_score': analysis2['flesch_reading_ease'],
                'grade_level': analysis2['flesch_kincaid_grade'],
                'readability_level': analysis2['readability_level']
            },
            'difference': {
                'flesch_score_diff': analysis1['flesch_reading_ease'] - analysis2['flesch_reading_ease'],
                'grade_level_diff': analysis1['flesch_kincaid_grade'] - analysis2['flesch_kincaid_grade']
            }
        }
        
        return comparison

# Uso en aplicación
analyzer = TextReadabilityAnalyzer()

# Analizar legibilidad
text = "This is a sample text for readability analysis. It contains multiple sentences to demonstrate various readability metrics."
analysis = analyzer.analyze_readability(text)
print(f"Readability analysis: {analysis}")

# Análisis en lote
texts = [
    "Simple text.",
    "This is a more complex text with multiple sentences and longer words.",
    "The utilization of sophisticated vocabulary and intricate sentence structures significantly enhances the complexity of textual content."
]

results_df = analyzer.analyze_texts_batch(texts)
print(results_df)

# Resumen de legibilidad
summary = analyzer.get_readability_summary(texts)
print(f"\\nReadability summary: {summary}")

# Comparar legibilidad
comparison = analyzer.compare_readability(texts[0], texts[2])
print(f"\\nReadability comparison: {comparison}")
""",
                alternatives=["readability", "nltk", "spacy"],
                documentation="https://github.com/shivam5992/textstat",
                github="https://github.com/shivam5992/textstat",
                pypi="https://pypi.org/project/textstat/"
            )
        }
    
    def get_library(self, name: str) -> LibraryInfo:
        """Obtener información de una librería específica."""
        return self.libraries.get(name)
    
    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Obtener todas las librerías."""
        return self.libraries
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values()]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las librerías de IA/ML realistas."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                requirements.append(f"{package}=={lib.version}")
        
        return '\n'.join(requirements)
    
    def get_realistic_usage_examples(self) -> Dict[str, str]:
        """Obtener ejemplos de uso real para cada librería."""
        return {name: lib.real_usage for name, lib in self.libraries.items()}




