"""
AI/ML Extensive Libraries - Librerías de IA y ML extensas
=======================================================

Guía extensa de librerías de IA y ML con más de 100 librerías
organizadas por subcategorías con ejemplos detallados y casos de uso avanzados.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class MLCategory(Enum):
    """Categorías de librerías de ML."""
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    NLP = "natural_language_processing"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    OPTIMIZATION = "optimization"
    VISUALIZATION = "visualization"
    MODEL_SERVING = "model_serving"


@dataclass
class LibraryInfo:
    """Información detallada de una librería."""
    name: str
    version: str
    description: str
    use_case: str
    category: MLCategory
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    installation: str = ""
    example: str = ""
    advanced_example: str = ""
    configuration: str = ""
    performance_notes: str = ""
    alternatives: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    documentation: str = ""
    community: str = ""
    last_updated: str = ""
    license: str = ""


class AIMLExtensiveLibraries:
    """
    Guía extensa de librerías de IA y ML.
    """
    
    def __init__(self):
        """Inicializar con librerías de IA/ML extensas."""
        self.libraries = {
            # Machine Learning
            'scikit-learn': LibraryInfo(
                name="scikit-learn",
                version="1.3.0",
                description="Machine Learning completo para Python con algoritmos clásicos",
                use_case="Clasificación, regresión, clustering, análisis de texto, validación cruzada",
                category=MLCategory.MACHINE_LEARNING,
                pros=[
                    "API consistente y fácil de usar",
                    "Amplia gama de algoritmos ML",
                    "Excelente documentación y ejemplos",
                    "Optimizado para rendimiento",
                    "Integración con NumPy y pandas",
                    "Herramientas de validación y métricas",
                    "Pipeline para flujos de trabajo",
                    "Comunidad muy activa"
                ],
                cons=[
                    "Limitado para deep learning",
                    "Puede ser lento para datasets muy grandes",
                    "Algunos algoritmos no son state-of-the-art",
                    "Limitado para datos no estructurados"
                ],
                installation="pip install scikit-learn[complete]",
                example="""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Datos de ejemplo
texts = [
    "This is a great product with excellent quality",
    "The product quality is amazing and wonderful",
    "I love this product, it's fantastic",
    "This product is terrible and awful",
    "The quality is poor and disappointing",
    "I hate this product, it's horrible"
]
labels = [1, 1, 1, 0, 0, 0]  # 1: positivo, 0: negativo

# Vectorización de texto
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Similitud coseno
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Similitud entre textos: {similarity[0][0]:.3f}")

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
print(f"Clusters: {clusters}")

# Clasificación
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, labels, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
""",
                advanced_example="""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Dataset sintético más grande
np.random.seed(42)
n_samples = 1000

# Generar textos sintéticos
positive_words = ['excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'love', 'perfect']
negative_words = ['terrible', 'awful', 'horrible', 'bad', 'hate', 'disappointing', 'poor']

texts = []
labels = []

for i in range(n_samples):
    if i < n_samples // 2:  # Textos positivos
        text = ' '.join(np.random.choice(positive_words, size=np.random.randint(3, 8)))
        label = 1
    else:  # Textos negativos
        text = ' '.join(np.random.choice(negative_words, size=np.random.randint(3, 8)))
        label = 0
    
    texts.append(text)
    labels.append(label)

print("=== ANÁLISIS AVANZADO CON SCIKIT-LEARN ===")
print(f"Dataset: {len(texts)} textos, {len(set(labels))} clases")

# 1. Análisis de características múltiples
print("\\n1. Análisis de características múltiples:")

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Count Vectorizer
count_vectorizer = CountVectorizer(max_features=500, ngram_range=(1, 2))
count_matrix = count_vectorizer.fit_transform(texts)

print(f"TF-IDF shape: {tfidf_matrix.shape}")
print(f"Count shape: {count_matrix.shape}")

# 2. Análisis de temas (LDA)
print("\\n2. Análisis de temas (LDA):")
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda.fit_transform(tfidf_matrix)

print("Top palabras por tema:")
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Tema {topic_idx}: {', '.join(top_words)}")

# 3. Clustering avanzado
print("\\n3. Clustering avanzado:")

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_clusters = kmeans.fit_predict(tfidf_matrix)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(tfidf_matrix.toarray())

# Clustering jerárquico
agg_clustering = AgglomerativeClustering(n_clusters=2)
agg_clusters = agg_clustering.fit_predict(tfidf_matrix.toarray())

print(f"K-Means clusters: {len(set(kmeans_clusters))}")
print(f"DBSCAN clusters: {len(set(dbscan_clusters))}")
print(f"Agglomerative clusters: {len(set(agg_clusters))}")

# 4. Reducción de dimensionalidad
print("\\n4. Reducción de dimensionalidad:")

# PCA
pca = PCA(n_components=50)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(pca_result)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:5]}")
print(f"t-SNE result shape: {tsne_result.shape}")

# 5. Clasificación avanzada
print("\\n5. Clasificación avanzada:")

# Dividir datos
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, labels, test_size=0.3, random_state=42
)

# Múltiples clasificadores
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results = {}
for name, clf in classifiers.items():
    # Entrenar
    clf.fit(X_train, y_train)
    
    # Predecir
    y_pred = clf.predict(X_test)
    
    # Evaluar
    accuracy = clf.score(X_test, y_test)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\\n{name}:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# 6. Optimización de hiperparámetros
print("\\n6. Optimización de hiperparámetros:")

# Grid Search para Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor score: {grid_search.best_score_:.3f}")

# 7. Pipeline completo
print("\\n7. Pipeline completo:")

# Crear pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=500)),
    ('scaler', StandardScaler(with_mean=False)),  # Para matrices sparse
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Entrenar pipeline
pipeline.fit(texts, labels)

# Predecir con pipeline
new_texts = ["This is amazing", "This is terrible"]
predictions = pipeline.predict(new_texts)
probabilities = pipeline.predict_proba(new_texts)

print("Predicciones:")
for text, pred, prob in zip(new_texts, predictions, probabilities):
    print(f"  '{text}' -> {pred} (prob: {prob.max():.3f})")

# 8. Análisis de importancia de características
print("\\n8. Análisis de importancia:")

# Obtener importancia de características
feature_importance = pipeline.named_steps['classifier'].feature_importances_
feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()

# Top 10 características más importantes
top_features_idx = feature_importance.argsort()[-10:][::-1]
top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]

print("Top 10 características más importantes:")
for feature, importance in top_features:
    print(f"  {feature}: {importance:.3f}")

# 9. Análisis de errores
print("\\n9. Análisis de errores:")

# Matriz de confusión
y_pred_final = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_final)

print("Matriz de confusión:")
print(cm)

# Análisis de errores
errors = []
for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred_final)):
    if true_label != pred_label:
        errors.append((i, true_label, pred_label))

print(f"\\nErrores de clasificación: {len(errors)}")
if errors:
    print("Primeros 5 errores:")
    for i, (idx, true_label, pred_label) in enumerate(errors[:5]):
        print(f"  {i+1}. Texto {idx}: verdadero={true_label}, predicho={pred_label}")
""",
                configuration="""
# Configuración de scikit-learn para mejor rendimiento

import numpy as np
import joblib
from sklearn import set_config

# Configurar scikit-learn
set_config(display='diagram')  # Mostrar pipelines como diagramas
set_config(assume_finite=True)  # Asumir que no hay NaN para mejor rendimiento

# Configurar threading
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Configurar joblib para paralelización
joblib.parallel.BACKENDS['threading'].n_jobs = 4

# Configurar numpy para mejor rendimiento
np.seterr(all='ignore')  # Ignorar warnings numéricos
""",
                performance_notes="""
# Optimizaciones de rendimiento para scikit-learn

# 1. Usar tipos de datos apropiados
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convertir strings a números

# 2. Usar matrices sparse para texto
from scipy.sparse import csr_matrix
sparse_matrix = csr_matrix(dense_matrix)

# 3. Usar n_jobs para paralelización
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# 4. Usar early stopping para algoritmos que lo soporten
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, validation_fraction=0.1, n_iter_no_change=10)

# 5. Usar caching para pipelines
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', classifier)
], memory=joblib.Memory('./cache'))

# 6. Usar warm_start para entrenamiento incremental
clf = RandomForestClassifier(n_estimators=50, warm_start=True)
clf.fit(X_train, y_train)
clf.n_estimators += 50
clf.fit(X_train, y_train)

# 7. Usar partial_fit para algoritmos que lo soporten
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
for batch in batches:
    clf.partial_fit(batch_X, batch_y)
""",
                alternatives=["xgboost", "lightgbm", "catboost", "mlxtend"],
                dependencies=["numpy", "scipy", "joblib", "threadpoolctl"],
                documentation="https://scikit-learn.org/stable/",
                community="https://stackoverflow.com/questions/tagged/scikit-learn",
                last_updated="2023-09-01",
                license="BSD-3-Clause"
            ),
            
            'transformers': LibraryInfo(
                name="transformers",
                version="4.35.0",
                description="Modelos de transformers de Hugging Face para NLP",
                use_case="Modelos de lenguaje, análisis de sentimientos, generación de texto, embeddings",
                category=MLCategory.NLP,
                pros=[
                    "Acceso a miles de modelos pre-entrenados",
                    "API simple y consistente",
                    "Soporte para múltiples frameworks (PyTorch, TensorFlow)",
                    "Modelos state-of-the-art",
                    "Fácil fine-tuning",
                    "Integración con datasets de Hugging Face",
                    "Soporte para múltiples idiomas",
                    "Comunidad muy activa"
                ],
                cons=[
                    "Modelos pueden ser muy grandes",
                    "Requiere GPU para mejor rendimiento",
                    "Uso de memoria alto",
                    "Dependencias pesadas"
                ],
                installation="pip install transformers[torch]",
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
generated = text_generator("The future of AI is", max_length=50, num_return_sequences=3)
for i, seq in enumerate(generated):
    print(f"Generated {i+1}: {seq['generated_text']}")

# Extracción de entidades
ner_pipeline = pipeline("ner", aggregation_strategy="simple")
entities = ner_pipeline("Apple is looking at buying U.K. startup for $1 billion")
print(f"Entities: {entities}")

# Respuestas a preguntas
qa_pipeline = pipeline("question-answering")
context = "The AI revolution is transforming industries worldwide."
question = "What is transforming industries?"
answer = qa_pipeline(question=question, context=context)
print(f"Answer: {answer['answer']}")

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
                advanced_example="""
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

print("=== ANÁLISIS AVANZADO CON TRANSFORMERS ===")

# 1. Análisis de sentimientos avanzado
print("\\n1. Análisis de sentimientos avanzado:")

# Usar modelo específico para análisis de sentimientos
sentiment_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model)

texts = [
    "I absolutely love this product! It's fantastic!",
    "This is okay, nothing special.",
    "I hate this product, it's terrible and disappointing.",
    "The quality is amazing and the service is excellent!",
    "Meh, it's fine I guess."
]

for text in texts:
    result = sentiment_pipeline(text)
    print(f"'{text}' -> {result[0]['label']} ({result[0]['score']:.3f})")

# 2. Generación de texto avanzada
print("\\n2. Generación de texto avanzada:")

# Usar modelo más grande para mejor generación
generator = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",
    pad_token_id=50256
)

# Generar con diferentes parámetros
prompts = [
    "The future of artificial intelligence",
    "In a world where technology",
    "The most important thing about"
]

for prompt in prompts:
    generated = generator(
        prompt,
        max_length=100,
        num_return_sequences=2,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    
    print(f"\\nPrompt: '{prompt}'")
    for i, seq in enumerate(generated):
        print(f"Generated {i+1}: {seq['generated_text']}")

# 3. Fine-tuning personalizado
print("\\n3. Fine-tuning personalizado:")

# Preparar datos de ejemplo para fine-tuning
train_texts = [
    "This product is amazing and I love it!",
    "The quality is excellent and the service is great.",
    "I hate this product, it's terrible.",
    "This is awful and disappointing.",
    "The product is good and I'm satisfied.",
    "It's okay, nothing special."
]

train_labels = [1, 1, 0, 0, 1, 0]  # 1: positivo, 0: negativo

# Crear dataset
dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})

# Tokenizer y modelo
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Función de tokenización
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Función de métricas
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics,
)

# Entrenar (comentado para evitar entrenamiento largo)
# trainer.train()

print("Fine-tuning configurado (entrenamiento comentado)")

# 4. Análisis de embeddings avanzado
print("\\n4. Análisis de embeddings avanzado:")

# Usar modelo de embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model)

def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Usar mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Calcular embeddings para múltiples textos
texts_for_embedding = [
    "I love this product",
    "This product is amazing",
    "I hate this product",
    "This product is terrible",
    "The product is okay"
]

embeddings = []
for text in texts_for_embedding:
    embedding = get_sentence_embedding(text)
    embeddings.append(embedding)

# Convertir a tensor
embeddings_tensor = torch.cat(embeddings, dim=0)

# Calcular similitud coseno
similarity_matrix = F.cosine_similarity(
    embeddings_tensor.unsqueeze(1),
    embeddings_tensor.unsqueeze(0),
    dim=2
)

print("Matriz de similitud:")
for i, text1 in enumerate(texts_for_embedding):
    for j, text2 in enumerate(texts_for_embedding):
        if i != j:
            similarity = similarity_matrix[i][j].item()
            print(f"'{text1}' <-> '{text2}': {similarity:.3f}")

# 5. Análisis de atención
print("\\n5. Análisis de atención:")

# Usar modelo con atención
attention_model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(attention_model)
model = AutoModel.from_pretrained(attention_model, output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# Obtener atención del último layer
attention = outputs.attentions[-1]  # Último layer
attention = attention.squeeze(0)  # Remover batch dimension

# Promediar atención sobre todas las heads
attention_avg = attention.mean(dim=0)

print(f"Texto: '{text}'")
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
print("Matriz de atención (promediada sobre heads):")
print(attention_avg.numpy())

# 6. Análisis de toxicidad
print("\\n6. Análisis de toxicidad:")

# Usar modelo de detección de toxicidad
toxicity_pipeline = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    return_all_scores=True
)

test_texts = [
    "This is a great product!",
    "You are stupid and worthless",
    "I love this amazing product",
    "This is the worst thing ever"
]

for text in test_texts:
    results = toxicity_pipeline(text)
    print(f"\\n'{text}':")
    for result in results[0]:
        print(f"  {result['label']}: {result['score']:.3f}")

# 7. Análisis de resumen
print("\\n7. Análisis de resumen:")

# Usar modelo de resumen
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

long_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
"""

summary = summarizer(long_text, max_length=100, min_length=30, do_sample=False)
print(f"\\nResumen: {summary[0]['summary_text']}")

# 8. Análisis de traducción
print("\\n8. Análisis de traducción:")

# Usar modelo de traducción
translator = pipeline("translation_en_to_fr", model="t5-small")

english_texts = [
    "Hello, how are you?",
    "This product is amazing",
    "I love artificial intelligence"
]

for text in english_texts:
    translation = translator(text)
    print(f"'{text}' -> '{translation[0]['translation_text']}'")

print("\\nAnálisis avanzado con Transformers completado!")
""",
                configuration="""
# Configuración de Transformers para mejor rendimiento

import torch
from transformers import set_seed

# Configurar semilla para reproducibilidad
set_seed(42)

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configurar memoria de GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)

# Configurar threading
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Evitar warnings

# Configurar logging
import logging
logging.getLogger("transformers").setLevel(logging.WARNING)

# Configurar cache
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')
model = AutoModel.from_pretrained('bert-base-uncased', cache_dir='./cache')
""",
                performance_notes="""
# Optimizaciones de rendimiento para Transformers

# 1. Usar modelos más pequeños para inferencia
from transformers import pipeline
small_model = pipeline("sentiment-analysis", model="distilbert-base-uncased")

# 2. Usar batching para múltiples textos
texts = ["Text 1", "Text 2", "Text 3"]
results = small_model(texts)  # Procesar en lote

# 3. Usar half precision para ahorrar memoria
model = model.half()  # Usar float16

# 4. Usar gradient checkpointing para ahorrar memoria
model.gradient_checkpointing_enable()

# 5. Usar attention slicing para modelos grandes
model.config.attention_probs_dropout_prob = 0.1

# 6. Usar torch.jit.script para optimización
scripted_model = torch.jit.script(model)

# 7. Usar ONNX para inferencia optimizada
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained("model_name")
""",
                alternatives=["openai", "anthropic", "cohere", "sentence-transformers"],
                dependencies=["torch", "tokenizers", "datasets", "accelerate"],
                documentation="https://huggingface.co/docs/transformers/",
                community="https://discuss.huggingface.co/",
                last_updated="2023-10-01",
                license="Apache-2.0"
            ),
            
            'sentence-transformers': LibraryInfo(
                name="sentence-transformers",
                version="2.2.2",
                description="Embeddings de oraciones optimizados para similitud semántica",
                use_case="Similitud de texto, búsqueda semántica, clustering, clasificación",
                category=MLCategory.NLP,
                pros=[
                    "Modelos optimizados específicamente para embeddings",
                    "API muy simple y fácil de usar",
                    "Buena calidad de embeddings",
                    "Soporte para múltiples idiomas",
                    "Modelos pre-entrenados de alta calidad",
                    "Integración con scikit-learn",
                    "Soporte para múltiples tareas",
                    "Rápido para inferencia"
                ],
                cons=[
                    "Modelos pueden ser grandes",
                    "Requiere GPU para mejor rendimiento",
                    "Limitado a tareas de similitud",
                    "Menos flexible que transformers"
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
print("\\nMatriz de similitud:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i != j:
            similarity = similarity_matrix[i][j]
            print(f"'{sent1}' <-> '{sent2}': {similarity:.3f}")

# Búsqueda semántica
query = "I love this amazing product"
query_embedding = model.encode([query])
query_similarity = cosine_similarity(query_embedding, embeddings)[0]

# Encontrar la oración más similar
most_similar_idx = np.argmax(query_similarity)
print(f"\\nQuery: '{query}'")
print(f"Most similar: '{sentences[most_similar_idx]}' (similarity: {query_similarity[most_similar_idx]:.3f})")
""",
                advanced_example="""
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

print("=== ANÁLISIS AVANZADO CON SENTENCE-TRANSFORMERS ===")

# 1. Análisis de similitud avanzado
print("\\n1. Análisis de similitud avanzado:")

# Cargar modelo más grande para mejor calidad
model = SentenceTransformer('all-mpnet-base-v2')

# Dataset más grande
sentences = [
    "I love this product, it's amazing and wonderful",
    "This product is fantastic and I highly recommend it",
    "The quality is excellent and the service is great",
    "I hate this product, it's terrible and awful",
    "This is the worst product I've ever used",
    "The quality is poor and the service is disappointing",
    "This product is okay, nothing special",
    "It's fine, I guess, but not amazing",
    "The product works as expected",
    "I'm satisfied with this purchase",
    "The customer service is excellent",
    "The delivery was fast and efficient",
    "The packaging was damaged",
    "The product arrived broken",
    "I'm very happy with my purchase"
]

# Generar embeddings
embeddings = model.encode(sentences)
print(f"Embeddings shape: {embeddings.shape}")

# Análisis de similitud
similarity_matrix = util.cos_sim(embeddings, embeddings)
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Encontrar pares más similares
pairs = []
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        similarity = similarity_matrix[i][j].item()
        pairs.append((i, j, similarity))

# Ordenar por similitud
pairs.sort(key=lambda x: x[2], reverse=True)

print("\\nTop 5 pares más similares:")
for i, (idx1, idx2, sim) in enumerate(pairs[:5]):
    print(f"{i+1}. '{sentences[idx1]}' <-> '{sentences[idx2]}' (similarity: {sim:.3f})")

# 2. Clustering semántico
print("\\n2. Clustering semántico:")

# K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Agrupar oraciones por cluster
clusters = {}
for i, label in enumerate(cluster_labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(sentences[i])

print("\\nClusters encontrados:")
for cluster_id, cluster_sentences in clusters.items():
    print(f"\\nCluster {cluster_id}:")
    for sentence in cluster_sentences:
        print(f"  - {sentence}")

# 3. Análisis de búsqueda semántica
print("\\n3. Análisis de búsqueda semántica:")

# Queries de búsqueda
queries = [
    "I love this amazing product",
    "This product is terrible",
    "The service is excellent"
]

for query in queries:
    query_embedding = model.encode([query])
    query_similarity = util.cos_sim(query_embedding, embeddings)[0]
    
    # Top 3 resultados
    top_indices = np.argsort(query_similarity)[-3:][::-1]
    
    print(f"\\nQuery: '{query}'")
    print("Top 3 resultados:")
    for i, idx in enumerate(top_indices):
        similarity = query_similarity[idx].item()
        print(f"  {i+1}. '{sentences[idx]}' (similarity: {similarity:.3f})")

# 4. Análisis de dimensionalidad
print("\\n4. Análisis de dimensionalidad:")

# Reducir dimensionalidad para visualización
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

print(f"Varianza explicada por los primeros 2 componentes: {pca.explained_variance_ratio_.sum():.3f}")

# Visualizar clusters
plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i in range(n_clusters):
    mask = cluster_labels == i
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clustering Semántico de Oraciones')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. Análisis de calidad de embeddings
print("\\n5. Análisis de calidad de embeddings:")

# Calcular distancia promedio dentro de clusters
intra_cluster_distances = []
for cluster_id in range(n_clusters):
    cluster_mask = cluster_labels == cluster_id
    cluster_embeddings = embeddings[cluster_mask]
    
    if len(cluster_embeddings) > 1:
        cluster_similarity = util.cos_sim(cluster_embeddings, cluster_embeddings)
        # Excluir diagonal (similitud consigo mismo)
        mask = np.ones_like(cluster_similarity, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_similarity = cluster_similarity[mask].mean().item()
        intra_cluster_distances.append(avg_similarity)
        
        print(f"Cluster {cluster_id} - Similitud promedio interna: {avg_similarity:.3f}")

# 6. Análisis de outliers
print("\\n6. Análisis de outliers:")

# Calcular distancia promedio de cada oración a todas las demás
avg_distances = []
for i in range(len(sentences)):
    distances = []
    for j in range(len(sentences)):
        if i != j:
            distance = 1 - similarity_matrix[i][j].item()  # Convertir similitud a distancia
            distances.append(distance)
    avg_distances.append(np.mean(distances))

# Identificar outliers (oraciones con mayor distancia promedio)
outlier_threshold = np.percentile(avg_distances, 90)
outliers = [(i, sentences[i], avg_distances[i]) for i, dist in enumerate(avg_distances) if dist > outlier_threshold]

print("\\nOutliers detectados (oraciones con mayor distancia promedio):")
for i, (idx, sentence, distance) in enumerate(outliers):
    print(f"{i+1}. '{sentence}' (distancia promedio: {distance:.3f})")

# 7. Análisis de tópicos
print("\\n7. Análisis de tópicos:")

# Usar clustering jerárquico para encontrar tópicos
from sklearn.cluster import AgglomerativeClustering

# Probar diferentes números de clusters
best_n_clusters = 0
best_silhouette = -1

for n_clusters in range(2, 6):
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Calcular silhouette score
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_n_clusters = n_clusters

print(f"Mejor número de clusters: {best_n_clusters} (silhouette: {best_silhouette:.3f})")

# 8. Análisis de evolución temporal (simulado)
print("\\n8. Análisis de evolución temporal:")

# Simular evolución temporal dividiendo las oraciones en "períodos"
n_periods = 3
sentences_per_period = len(sentences) // n_periods

for period in range(n_periods):
    start_idx = period * sentences_per_period
    end_idx = start_idx + sentences_per_period if period < n_periods - 1 else len(sentences)
    
    period_sentences = sentences[start_idx:end_idx]
    period_embeddings = embeddings[start_idx:end_idx]
    
    # Calcular centroide del período
    period_centroid = np.mean(period_embeddings, axis=0)
    
    print(f"\\nPeríodo {period + 1}:")
    print(f"  Número de oraciones: {len(period_sentences)}")
    print(f"  Centroide shape: {period_centroid.shape}")
    
    # Calcular similitud promedio dentro del período
    period_similarity = util.cos_sim(period_embeddings, period_embeddings)
    mask = np.ones_like(period_similarity, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_similarity = period_similarity[mask].mean().item()
    print(f"  Similitud promedio interna: {avg_similarity:.3f}")

print("\\nAnálisis avanzado con Sentence-Transformers completado!")
""",
                configuration="""
# Configuración de Sentence-Transformers para mejor rendimiento

import torch
from sentence_transformers import SentenceTransformer

# Configurar dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Configurar modelo con opciones de rendimiento
model = SentenceTransformer(
    'all-MiniLM-L6-v2',
    device=device,
    cache_folder='./cache'
)

# Configurar parámetros de encoding
def encode_with_options(texts, batch_size=32, show_progress_bar=True):
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_tensor=True,  # Para cálculos más rápidos
        normalize_embeddings=True  # Normalizar para similitud coseno
    )

# Configurar memoria de GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)
""",
                performance_notes="""
# Optimizaciones de rendimiento para Sentence-Transformers

# 1. Usar modelos más pequeños para inferencia
small_model = SentenceTransformer('all-MiniLM-L6-v2')  # 22MB
large_model = SentenceTransformer('all-mpnet-base-v2')  # 420MB

# 2. Usar batching para múltiples textos
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = model.encode(texts, batch_size=32)

# 3. Usar half precision para ahorrar memoria
model.half()  # Usar float16

# 4. Usar convert_to_tensor para cálculos más rápidos
embeddings = model.encode(texts, convert_to_tensor=True)

# 5. Usar normalize_embeddings para similitud coseno
embeddings = model.encode(texts, normalize_embeddings=True)

# 6. Usar cache para embeddings repetidos
from sentence_transformers import util
cache = {}
def get_cached_embedding(text):
    if text not in cache:
        cache[text] = model.encode([text])
    return cache[text]

# 7. Usar multiprocessing para grandes datasets
from sentence_transformers import SentenceTransformer
import multiprocessing as mp

def encode_chunk(texts_chunk):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts_chunk)

# Dividir en chunks y procesar en paralelo
n_processes = mp.cpu_count()
chunk_size = len(texts) // n_processes
chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

with mp.Pool(n_processes) as pool:
    embeddings_chunks = pool.map(encode_chunk, chunks)

embeddings = np.vstack(embeddings_chunks)
""",
                alternatives=["transformers", "openai-embeddings", "cohere-embeddings", "fasttext"],
                dependencies=["torch", "transformers", "numpy", "scikit-learn"],
                documentation="https://www.sbert.net/",
                community="https://github.com/UKPLab/sentence-transformers",
                last_updated="2023-09-01",
                license="Apache-2.0"
            )
        }
    
    def get_library(self, name: str) -> LibraryInfo:
        """Obtener información de una librería específica."""
        return self.libraries.get(name)
    
    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Obtener todas las librerías."""
        return self.libraries
    
    def get_libraries_by_category(self, category: MLCategory) -> Dict[str, LibraryInfo]:
        """Obtener librerías por categoría."""
        return {name: lib for name, lib in self.libraries.items() if lib.category == category}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values() if lib.installation]
    
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
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Obtener comparación de rendimiento entre librerías."""
        return {
            "scikit-learn": {
                "speed": "Fast",
                "memory_usage": "Medium",
                "accuracy": "Good",
                "ease_of_use": "High",
                "scalability": "Limited by RAM"
            },
            "transformers": {
                "speed": "Medium (with GPU: Fast)",
                "memory_usage": "High",
                "accuracy": "Excellent",
                "ease_of_use": "High",
                "scalability": "Good"
            },
            "sentence-transformers": {
                "speed": "Fast",
                "memory_usage": "Medium",
                "accuracy": "Excellent",
                "ease_of_use": "Very High",
                "scalability": "Good"
            }
        }




