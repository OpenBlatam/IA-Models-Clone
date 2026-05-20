"""
Sistema de análisis de fotos con diferentes dispositivos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class DeviceAnalysis:
    """Análisis con dispositivo específico"""
    device_type: str  # "smartphone", "tablet", "camera", "dermascope", "microscope"
    device_model: Optional[str] = None
    image_url: str = ""
    analysis_capabilities: List[str] = None
    limitations: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.analysis_capabilities is None:
            self.analysis_capabilities = []
        if self.limitations is None:
            self.limitations = []
        if self.recommendations is None:
            self.recommendations = []
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "device_type": self.device_type,
            "device_model": self.device_model,
            "image_url": self.image_url,
            "analysis_capabilities": self.analysis_capabilities,
            "limitations": self.limitations,
            "recommendations": self.recommendations
        }


@dataclass
class DeviceReport:
    """Reporte de análisis de dispositivos"""
    id: str
    user_id: str
    analyses: List[DeviceAnalysis]
    optimal_device: str
    device_comparison: Dict
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "analyses": [a.to_dict() for a in self.analyses],
            "optimal_device": self.optimal_device,
            "device_comparison": self.device_comparison,
            "created_at": self.created_at
        }


class DeviceAnalysisSystem:
    """Sistema de análisis con diferentes dispositivos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reports: Dict[str, List[DeviceReport]] = {}  # user_id -> [reports]
    
    def analyze_with_device(self, user_id: str, device_data: List[Dict]) -> DeviceReport:
        """Analiza con diferentes dispositivos"""
        analyses = []
        
        for dev_data in device_data:
            device_type = dev_data.get("device_type", "smartphone")
            device_model = dev_data.get("device_model")
            image_url = dev_data.get("image_url", "")
            
            # Capacidades según dispositivo
            capabilities, limitations = self._get_device_capabilities(device_type)
            
            # Recomendaciones
            recommendations = self._generate_device_recommendations(device_type)
            
            analysis = DeviceAnalysis(
                device_type=device_type,
                device_model=device_model,
                image_url=image_url,
                analysis_capabilities=capabilities,
                limitations=limitations,
                recommendations=recommendations
            )
            analyses.append(analysis)
        
        # Determinar dispositivo óptimo
        optimal_device = "dermascope"  # Por defecto, el más preciso
        
        # Comparación de dispositivos
        device_comparison = {
            "best_for_analysis": "dermascope",
            "most_accessible": "smartphone",
            "note": "Dispositivos especializados ofrecen mejor precisión"
        }
        
        report = DeviceReport(
            id=str(uuid.uuid4()),
            user_id=user_id,
            analyses=analyses,
            optimal_device=optimal_device,
            device_comparison=device_comparison
        )
        
        if user_id not in self.reports:
            self.reports[user_id] = []
        
        self.reports[user_id].append(report)
        return report
    
    def _get_device_capabilities(self, device_type: str) -> tuple:
        """Obtiene capacidades y limitaciones según dispositivo"""
        capabilities_map = {
            "smartphone": (
                ["Análisis básico", "Detección de tono", "Textura general"],
                ["Resolución limitada", "Sin aumento", "Iluminación variable"]
            ),
            "tablet": (
                ["Análisis básico", "Pantalla grande", "Mejor visualización"],
                ["Similar a smartphone", "Sin aumento especializado"]
            ),
            "camera": (
                ["Alta resolución", "Control de iluminación", "Análisis detallado"],
                ["Requiere configuración", "No especializado para piel"]
            ),
            "dermascope": (
                ["Análisis profesional", "Aumento especializado", "Iluminación controlada", "Máxima precisión"],
                ["Costo elevado", "Requiere entrenamiento"]
            ),
            "microscope": (
                ["Análisis microscópico", "Máximo detalle", "Estructura celular"],
                ["Muy costoso", "Requiere experto", "No práctico para uso diario"]
            )
        }
        return capabilities_map.get(device_type.lower(), capabilities_map["smartphone"])
    
    def _generate_device_recommendations(self, device_type: str) -> List[str]:
        """Genera recomendaciones según dispositivo"""
        recommendations = []
        
        if device_type == "smartphone":
            recommendations.append("Usa buena iluminación natural")
            recommendations.append("Mantén el dispositivo estable")
            recommendations.append("Considera un accesorio de aumento para mejor detalle")
        elif device_type == "dermascope":
            recommendations.append("Dispositivo ideal para análisis profesional")
            recommendations.append("Sigue protocolos de limpieza del dispositivo")
        elif device_type == "camera":
            recommendations.append("Configura iluminación consistente")
            recommendations.append("Usa trípode para estabilidad")
        
        return recommendations
    
    def get_user_reports(self, user_id: str) -> List[DeviceReport]:
        """Obtiene reportes del usuario"""
        return self.reports.get(user_id, [])






