"""
Sistema de análisis avanzado de textura
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class TextureAnalysis:
    """Análisis de textura"""
    id: str
    user_id: str
    image_url: str
    smoothness_score: float  # 0-100
    roughness_score: float  # 0-100
    pore_density: float
    pore_size_average: float
    wrinkle_density: float
    texture_uniformity: float
    recommendations: List[str]
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "image_url": self.image_url,
            "smoothness_score": self.smoothness_score,
            "roughness_score": self.roughness_score,
            "pore_density": self.pore_density,
            "pore_size_average": self.pore_size_average,
            "wrinkle_density": self.wrinkle_density,
            "texture_uniformity": self.texture_uniformity,
            "recommendations": self.recommendations,
            "created_at": self.created_at
        }


class AdvancedTextureAnalysis:
    """Sistema de análisis avanzado de textura"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.analyses: Dict[str, List[TextureAnalysis]] = {}  # user_id -> [analyses]
    
    def analyze_texture(self, user_id: str, image_url: str) -> TextureAnalysis:
        """Analiza textura de la piel"""
        # Simulación de análisis avanzado
        # En producción usaría procesamiento de imagen real
        
        smoothness_score = 75.0
        roughness_score = 25.0
        pore_density = 0.15
        pore_size_average = 0.08
        wrinkle_density = 0.12
        texture_uniformity = 0.70
        
        recommendations = []
        
        if roughness_score > 40:
            recommendations.append("Textura rugosa detectada. Usa exfoliantes suaves.")
        
        if pore_density > 0.20:
            recommendations.append("Poros visibles. Considera productos para minimizar poros.")
        
        if wrinkle_density > 0.15:
            recommendations.append("Arrugas detectadas. Considera productos anti-envejecimiento.")
        
        if texture_uniformity < 0.60:
            recommendations.append("Textura irregular. Usa productos para uniformizar.")
        
        if not recommendations:
            recommendations.append("Textura de piel saludable. Mantén tu rutina actual.")
        
        analysis = TextureAnalysis(
            id=str(uuid.uuid4()),
            user_id=user_id,
            image_url=image_url,
            smoothness_score=smoothness_score,
            roughness_score=roughness_score,
            pore_density=pore_density,
            pore_size_average=pore_size_average,
            wrinkle_density=wrinkle_density,
            texture_uniformity=texture_uniformity,
            recommendations=recommendations
        )
        
        if user_id not in self.analyses:
            self.analyses[user_id] = []
        
        self.analyses[user_id].append(analysis)
        return analysis
    
    def compare_textures(self, user_id: str, analysis1_id: str,
                        analysis2_id: str) -> Dict:
        """Compara dos análisis de textura"""
        user_analyses = self.analyses.get(user_id, [])
        
        analysis1 = next((a for a in user_analyses if a.id == analysis1_id), None)
        analysis2 = next((a for a in user_analyses if a.id == analysis2_id), None)
        
        if not analysis1 or not analysis2:
            raise ValueError("One or both analyses not found")
        
        # Calcular cambios
        smoothness_change = analysis2.smoothness_score - analysis1.smoothness_score
        roughness_change = analysis2.roughness_score - analysis1.roughness_score
        pore_change = analysis2.pore_density - analysis1.pore_density
        
        improvements = []
        if smoothness_change > 5:
            improvements.append(f"Suavidad mejoró {smoothness_change:.1f}%")
        if roughness_change < -5:
            improvements.append(f"Rugosidad disminuyó {abs(roughness_change):.1f}%")
        if pore_change < -0.05:
            improvements.append("Densidad de poros mejoró")
        
        return {
            "analysis1": analysis1.to_dict(),
            "analysis2": analysis2.to_dict(),
            "smoothness_change": smoothness_change,
            "roughness_change": roughness_change,
            "pore_change": pore_change,
            "improvements": improvements,
            "overall_improvement": len(improvements) > 0
        }
    
    def get_user_analyses(self, user_id: str) -> List[TextureAnalysis]:
        """Obtiene análisis del usuario"""
        return self.analyses.get(user_id, [])






