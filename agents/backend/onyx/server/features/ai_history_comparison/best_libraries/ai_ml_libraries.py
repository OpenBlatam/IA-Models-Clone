"""
AI/ML Libraries - Mejores librerías de IA y Machine Learning
==========================================================

Las mejores librerías para funcionalidades de IA y ML.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class LibraryInfo:
    """Información de una librería."""
    name: str
    version: str
    description: str
    use_case: str
    pros: List[str]
    cons: List[str]
    installation: str
    example: str


class AIMLLibraries:
    """
    Mejores librerías para IA y Machine Learning.
    """
    
    def __init__(self):
        """Inicializar con las mejores librerías de IA/ML."""
        self.libraries = {
            # Machine Learning
            'scikit-learn': LibraryInfo(
                name="scikit-learn",
                version="1.3.0",
                description="Machine Learning para Python",
                use_case="Clasificación, regresión, clustering, análisis de texto",
                pros=[
                    "API consistente y fácil de usar",
                    "Amplia gama de algoritmos",
                    "Excelente documentación",
                    "Optimizado para rendimiento"
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

# Vectorización de texto
vectorizer = TfidfVectorizer()
texts = ["Texto 1", "Texto 2", "Texto 3"]
tfidf_matrix = vectorizer.fit_transform(texts)

# Similitud coseno
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

# Clustering
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(tfidf_matrix)
"""
            ),
            
            'transformers': LibraryInfo(
                name="transformers",
                version="4.35.0",
                description="Modelos de transformers de Hugging Face",
                use_case="Modelos de lenguaje, análisis de sentimientos, generación de texto",
                pros=[
                    "Acceso a miles de modelos pre-entrenados",
                    "Fácil de usar",
                    "Soporte para múltiples frameworks",
                    "Modelos state-of-the-art"
                ],
                cons=[
                    "Modelos pueden ser grandes",
                    "Requiere GPU para mejor rendimiento"
                ],
                installation="pip install transformers torch",
                example="""
from transformers import pipeline

# Análisis de sentimientos
sentiment_analyzer = pipeline("sentiment-analysis")
result = sentiment_analyzer("I love this product!")

# Generación de texto
text_generator = pipeline("text-generation", model="gpt2")
generated = text_generator("The future of AI is", max_length=50)
"""
            ),
            
            'openai': LibraryInfo(
                name="openai",
                version="1.3.0",
                description="API de OpenAI para GPT y otros modelos",
                use_case="Generación de texto, análisis, embeddings",
                pros=[
                    "Acceso a GPT-4 y otros modelos avanzados",
                    "API simple y bien documentada",
                    "Modelos de alta calidad",
                    "Soporte para embeddings"
                ],
                cons=[
                    "Requiere API key",
                    "Costo por uso",
                    "Dependencia de internet"
                ],
                installation="pip install openai",
                example="""
import openai

# Configurar API key
openai.api_key = "your-api-key"

# Generar texto
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain AI in simple terms"}]
)

# Obtener embeddings
embeddings = openai.Embedding.create(
    input="Your text here",
    model="text-embedding-ada-002"
)
"""
            ),
            
            'sentence-transformers': LibraryInfo(
                name="sentence-transformers",
                version="2.2.2",
                description="Embeddings de oraciones",
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

# Cargar modelo
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generar embeddings
sentences = ["This is a test", "This is another test"]
embeddings = model.encode(sentences)

# Calcular similitud
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
"""
            ),
            
            # Text Analysis
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
corrected = text.correct()
print(f"Corrected: {corrected}")
"""
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
text = "This product is amazing! 😍"
scores = analyzer.polarity_scores(text)
print(f"Scores: {scores}")
"""
            ),
            
            # Similarity and Clustering
            'gensim': LibraryInfo(
                name="gensim",
                version="4.3.0",
                description="Modelado de temas y similitud de documentos",
                use_case="LDA, Word2Vec, similitud de documentos, modelado de temas",
                pros=[
                    "Excelente para modelado de temas",
                    "Implementaciones eficientes",
                    "Bueno para grandes datasets",
                    "Soporte para Word2Vec y Doc2Vec"
                ],
                cons=[
                    "Curva de aprendizaje",
                    "Limitado a ciertos tipos de análisis"
                ],
                installation="pip install gensim",
                example="""
from gensim import corpora, models, similarities
from gensim.models import Word2Vec

# Crear diccionario
texts = [['word1', 'word2'], ['word2', 'word3']]
dictionary = corpora.Dictionary(texts)

# Crear corpus
corpus = [dictionary.doc2bow(text) for text in texts]

# Modelo LDA
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary)

# Word2Vec
model = Word2Vec(texts, min_count=1)
similarity = model.wv.similarity('word1', 'word2')
"""
            ),
            
            # Deep Learning
            'torch': LibraryInfo(
                name="torch",
                version="2.1.0",
                description="Framework de deep learning",
                use_case="Redes neuronales, deep learning, modelos personalizados",
                pros=[
                    "Muy flexible",
                    "Excelente rendimiento",
                    "Amplia comunidad",
                    "Soporte para GPU"
                ],
                cons=[
                    "Curva de aprendizaje empinada",
                    "Requiere conocimiento de deep learning"
                ],
                installation="pip install torch",
                example="""
import torch
import torch.nn as nn

# Definir red neuronal simple
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Usar modelo
model = SimpleNN(10, 5, 1)
input_tensor = torch.randn(1, 10)
output = model(input_tensor)
"""
            ),
            
            # Data Visualization
            'matplotlib': LibraryInfo(
                name="matplotlib",
                version="3.7.0",
                description="Visualización de datos",
                use_case="Gráficos, visualizaciones, reportes",
                pros=[
                    "Muy flexible",
                    "Amplia gama de gráficos",
                    "Bien documentado",
                    "Estándar de la industria"
                ],
                cons=[
                    "API puede ser verbosa",
                    "No tan moderno como otras librerías"
                ],
                installation="pip install matplotlib",
                example="""
import matplotlib.pyplot as plt
import numpy as np

# Crear datos
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Crear gráfico
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
"""
            ),
            
            'seaborn': LibraryInfo(
                name="seaborn",
                version="0.12.0",
                description="Visualización estadística",
                use_case="Gráficos estadísticos, análisis exploratorio",
                pros=[
                    "Gráficos hermosos por defecto",
                    "Integración con pandas",
                    "Fácil de usar",
                    "Bueno para análisis estadístico"
                ],
                cons=[
                    "Menos flexible que matplotlib",
                    "Limitado a ciertos tipos de gráficos"
                ],
                installation="pip install seaborn",
                example="""
import seaborn as sns
import pandas as pd

# Crear datos
data = pd.DataFrame({
    'quality': [0.8, 0.9, 0.7, 0.6],
    'model': ['gpt-4', 'claude-3', 'gpt-4', 'claude-3']
})

# Crear gráfico
sns.boxplot(data=data, x='model', y='quality')
plt.title('Quality by Model')
plt.show()
"""
            )
        }
    
    def get_library(self, name: str) -> LibraryInfo:
        """Obtener información de una librería específica."""
        return self.libraries.get(name)
    
    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Obtener todas las librerías."""
        return self.libraries
    
    def get_libraries_by_category(self, category: str) -> Dict[str, LibraryInfo]:
        """Obtener librerías por categoría."""
        categories = {
            'machine_learning': ['scikit-learn', 'torch'],
            'nlp': ['transformers', 'textblob', 'vaderSentiment'],
            'embeddings': ['sentence-transformers', 'gensim'],
            'apis': ['openai'],
            'visualization': ['matplotlib', 'seaborn']
        }
        
        if category not in categories:
            return {}
        
        return {name: self.libraries[name] for name in categories[category] if name in self.libraries}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values()]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las mejores librerías de IA/ML."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                if '==' in package:
                    requirements.append(package)
                else:
                    requirements.append(f"{package}>=latest")
        
        return '\n'.join(requirements)




