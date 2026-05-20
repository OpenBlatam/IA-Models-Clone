"""
Network Analysis Service - Análisis de red profesional
=======================================================

Sistema de análisis de red profesional y networking.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Contact:
    """Contacto profesional"""
    id: str
    name: str
    title: str
    company: str
    industry: str
    connection_strength: float  # 0.0 - 1.0
    last_interaction: Optional[datetime] = None
    mutual_connections: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class NetworkInsight:
    """Insight de red"""
    total_contacts: int
    strong_connections: int
    weak_connections: int
    industries: Dict[str, int]
    companies: Dict[str, int]
    recommendations: List[str]
    network_score: float


class NetworkAnalysisService:
    """Servicio de análisis de red"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.networks: Dict[str, List[Contact]] = {}  # user_id -> contacts
        logger.info("NetworkAnalysisService initialized")
    
    def add_contact(
        self,
        user_id: str,
        name: str,
        title: str,
        company: str,
        industry: str,
        connection_strength: float = 0.5
    ) -> Contact:
        """Agregar contacto a la red"""
        contact_id = f"contact_{user_id}_{int(datetime.now().timestamp())}"
        
        contact = Contact(
            id=contact_id,
            name=name,
            title=title,
            company=company,
            industry=industry,
            connection_strength=connection_strength,
            last_interaction=datetime.now(),
        )
        
        if user_id not in self.networks:
            self.networks[user_id] = []
        
        self.networks[user_id].append(contact)
        
        logger.info(f"Contact added to network: {contact_id}")
        return contact
    
    def analyze_network(self, user_id: str) -> NetworkInsight:
        """Analizar red profesional"""
        contacts = self.networks.get(user_id, [])
        
        if not contacts:
            return NetworkInsight(
                total_contacts=0,
                strong_connections=0,
                weak_connections=0,
                industries={},
                companies={},
                recommendations=["Comienza a construir tu red profesional"],
                network_score=0.0,
            )
        
        # Calcular métricas
        strong_connections = sum(1 for c in contacts if c.connection_strength > 0.7)
        weak_connections = len(contacts) - strong_connections
        
        # Agrupar por industria
        industries = defaultdict(int)
        for contact in contacts:
            industries[contact.industry] += 1
        
        # Agrupar por empresa
        companies = defaultdict(int)
        for contact in contacts:
            companies[contact.company] += 1
        
        # Calcular score de red
        network_score = self._calculate_network_score(contacts)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(contacts, industries, companies)
        
        return NetworkInsight(
            total_contacts=len(contacts),
            strong_connections=strong_connections,
            weak_connections=weak_connections,
            industries=dict(industries),
            companies=dict(companies),
            recommendations=recommendations,
            network_score=network_score,
        )
    
    def _calculate_network_score(self, contacts: List[Contact]) -> float:
        """Calcular score de red"""
        if not contacts:
            return 0.0
        
        # Factor 1: Número de contactos (normalizado)
        size_score = min(1.0, len(contacts) / 100.0)
        
        # Factor 2: Fuerza promedio de conexiones
        avg_strength = sum(c.connection_strength for c in contacts) / len(contacts)
        
        # Factor 3: Diversidad de industrias
        industries = set(c.industry for c in contacts)
        diversity_score = min(1.0, len(industries) / 10.0)
        
        # Score combinado
        score = (size_score * 0.3 + avg_strength * 0.5 + diversity_score * 0.2)
        
        return round(score, 2)
    
    def _generate_recommendations(
        self,
        contacts: List[Contact],
        industries: Dict[str, int],
        companies: Dict[str, int]
    ) -> List[str]:
        """Generar recomendaciones de networking"""
        recommendations = []
        
        # Recomendación de diversidad
        if len(industries) < 3:
            recommendations.append(
                "Diversifica tu red: conecta con personas de diferentes industrias"
            )
        
        # Recomendación de conexiones fuertes
        strong = sum(1 for c in contacts if c.connection_strength > 0.7)
        if strong < len(contacts) * 0.3:
            recommendations.append(
                "Fortalece conexiones existentes: mantén contacto regular con tu red"
            )
        
        # Recomendación de tamaño
        if len(contacts) < 50:
            recommendations.append(
                "Expande tu red: apunta a tener al menos 50 contactos profesionales"
            )
        
        # Recomendación de industrias objetivo
        if not recommendations:
            recommendations.append(
                "Tu red está bien balanceada. Considera conectar con líderes en tu industria objetivo"
            )
        
        return recommendations
    
    def find_introductions(
        self,
        user_id: str,
        target_company: str,
        target_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Encontrar posibles introducciones"""
        contacts = self.networks.get(user_id, [])
        
        # Buscar contactos en la empresa objetivo
        direct_contacts = [
            c for c in contacts
            if target_company.lower() in c.company.lower()
        ]
        
        # Buscar contactos que puedan hacer introducción
        potential_introducers = [
            c for c in contacts
            if c.connection_strength > 0.6
            and c.company.lower() != target_company.lower()
        ]
        
        return [
            {
                "type": "direct",
                "contact": {
                    "name": c.name,
                    "title": c.title,
                    "company": c.company,
                    "connection_strength": c.connection_strength,
                }
            }
            for c in direct_contacts
        ] + [
            {
                "type": "introduction",
                "contact": {
                    "name": c.name,
                    "title": c.title,
                    "company": c.company,
                    "connection_strength": c.connection_strength,
                },
                "reason": f"Puede hacer introducción en {target_company}",
            }
            for c in potential_introducers[:5]
        ]
    
    def get_network_path(
        self,
        user_id: str,
        target_person: str
    ) -> Optional[List[str]]:
        """Encontrar camino en la red hacia una persona"""
        # En producción, esto usaría algoritmos de grafos (BFS/DFS)
        # Por ahora, simulamos
        contacts = self.networks.get(user_id, [])
        
        # Buscar contacto directo
        for contact in contacts:
            if target_person.lower() in contact.name.lower():
                return [contact.name]
        
        # Buscar conexión de segundo grado (simulado)
        return None




