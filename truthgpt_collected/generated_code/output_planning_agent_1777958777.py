# RAG pipeline improvement
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("trusted_knowledge.index")

def retrieve_and_verify(query, top_k=5):
    query_emb = model.encode([query])
    distances, indices = index.search(query_emb, top_k)
    # Filter by distance threshold to ensure relevance
    relevant = [docs[i] for i, d in zip(indices[0], distances[0]) if d < 0.6]
    return relevant