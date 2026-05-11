# truthgpt_selfverification.py - Integración del paper "Do I Really Know? Learning Factual Self-Verification for Hallucination Reduction"
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthSelfVerifier:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.eval()
        # Base de conocimiento simple (en producción usar una vector DB)
        self.knowledge_base = [
            "La Tierra gira alrededor del Sol.",
            "El agua hierve a 100°C al nivel del mar.",
            "París es la capital de Francia."
        ]
        self.uncertainty_threshold = 0.6  # umbral de confianza para declinar responder

    def retrieve_evidence(self, claim: str):
        """Recupera evidencia relevante de la base de conocimiento (simulación)."""
        for fact in self.knowledge_base:
            if any(word in claim.lower() for word in fact.lower().split()):
                return fact
        return None

    def verify(self, claim: str, return_confidence: bool = False):
        """Verifica una afirmación y devuelve predicción y confianza."""
        evidence = self.retrieve_evidence(claim)
        if evidence:
            input_text = f"Afirmación: {claim} Evidencia: {evidence}"
        else:
            input_text = claim
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
        result = {
            "claim": claim,
            "prediction": "verdadero" if predicted.item() == 1 else "falso",
            "confidence": confidence.item(),
            "evidence": evidence
        }
        if return_confidence:
            return result, confidence.item()
        return result

    def should_decline(self, claim: str) -> bool:
        """Decide si el modelo debe declinar responder por falta de confianza."""
        _, confidence = self.verify(claim, return_confidence=True)
        if confidence < self.uncertainty_threshold:
            logger.info(f"Declinando responder a: {claim} (confianza {confidence:.2f})")
            return True
        return False

class TruthGPT:
    def __init__(self):
        self.verifier = TruthSelfVerifier()
        self.history = []

    def respond(self, query: str) -> str:
        # Autoverificación: si el modelo no está seguro, declina
        if self.verifier.should_decline(query):
            response = "TruthGPT: No estoy seguro de la respuesta a esa pregunta. Prefiero no especular."
            self.history.append({"query": query, "response": response, "declined": True})
            return response
        # Verificación regular
        result = self.verifier.verify(query)
        response = f"TruthGPT: La afirmación '{query}' se considera {result['prediction']} con una confianza de {result['confidence']:.2f}"
        if result['evidence']:
            response += f" (Basado en evidencia: {result['evidence']})"
        self.history.append({"query": query, "response": response, "declined": False})
        return response

    def add_knowledge(self, fact: str):
        self.verifier.knowledge_base.append(fact)
        logger.info(f"Nuevo conocimiento añadido: {fact}")

# Ejemplo de uso
if __name__ == "__main__":
    agent = TruthGPT()
    print(agent.respond("La Tierra gira alrededor del Sol."))
    print(agent.respond("El helio es más pesado que el aire."))  # No en KB, baja confianza
    agent.add_knowledge("El helio es más ligero que el aire.")
    print(agent.respond("El helio es más pesado que el aire."))  # Ahora con evidencia
    print("Historial de consultas:", len(agent.history))