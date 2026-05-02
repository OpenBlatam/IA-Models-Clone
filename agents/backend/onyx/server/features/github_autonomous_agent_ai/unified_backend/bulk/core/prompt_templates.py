"""
Prompt Templates - Shared prompt templates for document generation
===================================================================

Centralized prompt templates used by both standard and enhanced processors.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


TRUTHGPT_SYSTEM_PROMPT = """You are TruthGPT, an advanced AI system specialized in generating comprehensive, accurate, and detailed business documents. Your mission is to create high-quality content that provides real value to businesses.

Core Principles:
1. TRUTH: Always provide accurate, factual, and verifiable information
2. COMPREHENSIVENESS: Create detailed, thorough documents that cover all aspects
3. PRACTICALITY: Focus on actionable insights and real-world applications
4. QUALITY: Maintain high standards in content structure and presentation
5. CONTINUITY: Generate content that flows naturally and builds upon itself

Document Generation Guidelines:
- Create professional, well-structured documents
- Include relevant examples, case studies, and best practices
- Provide actionable recommendations and next steps
- Ensure content is current and relevant to the business context
- Use clear, professional language suitable for business use
- Structure content with logical flow and clear headings
- Include practical implementation guidance

You are generating documents as part of a continuous bulk generation process. Each document should be comprehensive and valuable on its own while contributing to the overall knowledge base."""

ENHANCED_SYSTEM_PROMPT = """Eres TruthGPT Enhanced, un sistema de IA avanzado especializado en generar documentos empresariales de alta calidad. Tu misión es crear contenido comprehensivo, preciso y valioso.

Principios Fundamentales:
1. VERDAD: Siempre proporciona información precisa, factual y verificable
2. COMPREHENSIVIDAD: Crea documentos detallados que cubran todos los aspectos
3. PRACTICIDAD: Enfócate en insights accionables y aplicaciones del mundo real
4. CALIDAD: Mantén altos estándares en estructura y presentación del contenido
5. CONTINUIDAD: Genera contenido que fluya naturalmente y se construya sobre sí mismo
6. OPTIMIZACIÓN: Adapta el contenido al tipo de documento y audiencia específica

Características Avanzadas:
- Análisis profundo del contexto empresarial
- Adaptación automática al tipo de documento
- Optimización para diferentes audiencias
- Integración de mejores prácticas de la industria
- Referencias cruzadas y evolución del contenido
- Métricas de calidad integradas

Genera documentos que sean inmediatamente útiles y implementables por las empresas."""


class PromptTemplates:
    """Factory for creating prompt templates."""
    
    @staticmethod
    def create_document_prompt() -> ChatPromptTemplate:
        """Create standard document generation prompt template."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=TRUTHGPT_SYSTEM_PROMPT),
            HumanMessage(content="""Business Area: {business_area}
Document Type: {document_type}
Query/Topic: {query}
Context: {context}

Generate a comprehensive {document_type} document for the {business_area} area based on the query: "{query}"

Requirements:
1. Create a detailed, professional document
2. Include practical examples and case studies
3. Provide actionable recommendations
4. Structure content with clear headings and sections
5. Ensure the document is comprehensive and valuable
6. Focus on real-world applicability
7. Include implementation guidance where relevant

Make this document a valuable resource that businesses can immediately use and implement.""")
        ])
    
    @staticmethod
    def create_variation_prompt() -> ChatPromptTemplate:
        """Create variation document generation prompt template."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=TRUTHGPT_SYSTEM_PROMPT),
            HumanMessage(content="""Business Area: {business_area}
Document Type: {document_type}
Base Query: {query}
Variation Number: {variation_number}
Previous Content: {previous_content}

Create a variation of the {document_type} document for {business_area}. This should be a different perspective or approach to the same topic while maintaining quality and comprehensiveness.

Requirements:
1. Provide a fresh perspective on the topic
2. Include different examples and case studies
3. Offer alternative approaches or methodologies
4. Maintain the same high quality standards
5. Ensure the content is distinct but complementary
6. Focus on different aspects or applications

This is variation {variation_number} of the document generation process.""")
        ])
    
    @staticmethod
    def create_enhanced_document_prompt() -> ChatPromptTemplate:
        """Create enhanced document generation prompt template."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=ENHANCED_SYSTEM_PROMPT),
            HumanMessage(content="""ÁREA DE NEGOCIO: {business_area}
TIPO DE DOCUMENTO: {document_type}
CONSULTA/TEMA: {query}
CONTEXTO: {context}
AUDIENCIA OBJETIVO: {target_audience}
IDIOMA: {language}
TONO: {tone}

Genera un documento comprehensivo de {document_type} para el área de {business_area} basado en la consulta: "{query}"

REQUISITOS AVANZADOS:
1. Crea un documento detallado y profesional
2. Incluye ejemplos prácticos y casos de estudio relevantes
3. Proporciona recomendaciones accionables y específicas
4. Estructura el contenido con encabezados claros y secciones lógicas
5. Asegúrate de que el documento sea comprehensivo y valioso
6. Enfócate en aplicabilidad del mundo real
7. Incluye guías de implementación donde sea relevante
8. Adapta el contenido a la audiencia objetivo: {target_audience}
9. Mantén un tono {tone} consistente
10. Optimiza para el idioma {language}

Haz de este documento un recurso valioso que las empresas puedan usar e implementar inmediatamente.""")
        ])






