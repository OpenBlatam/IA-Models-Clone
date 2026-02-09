"""
Análisis Comparativo entre Usuarios
====================================
Comparación y benchmarking entre usuarios
"""

from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
import structlog
from collections import defaultdict

from .models import PsychologicalProfile, PsychologicalValidation

logger = structlog.get_logger()


class ComparativeAnalyzer:
    """Analizador comparativo"""
    
    def __init__(self):
        """Inicializar analizador"""
        logger.info("ComparativeAnalyzer initialized")
    
    def compare_users(
        self,
        profiles: List[PsychologicalProfile],
        user_ids: List[UUID]
    ) -> Dict[str, Any]:
        """
        Comparar múltiples usuarios
        
        Args:
            profiles: Lista de perfiles
            user_ids: IDs de usuarios
            
        Returns:
            Comparación detallada
        """
        if len(profiles) != len(user_ids):
            raise ValueError("Number of profiles must match number of user IDs")
        
        if len(profiles) < 2:
            raise ValueError("Need at least 2 users for comparison")
        
        # Agrupar por usuario
        user_profiles = {}
        for profile, user_id in zip(profiles, user_ids):
            if user_id not in user_profiles:
                user_profiles[user_id] = []
            user_profiles[user_id].append(profile)
        
        # Calcular promedios por usuario
        user_averages = {}
        for user_id, user_profiles_list in user_profiles.items():
            if not user_profiles_list:
                continue
            
            # Promediar rasgos de personalidad
            avg_traits = defaultdict(list)
            avg_confidence = []
            
            for profile in user_profiles_list:
                for trait, value in profile.personality_traits.items():
                    avg_traits[trait].append(value)
                avg_confidence.append(profile.confidence_score)
            
            user_averages[user_id] = {
                "personality_traits": {
                    trait: sum(values) / len(values)
                    for trait, values in avg_traits.items()
                },
                "average_confidence": sum(avg_confidence) / len(avg_confidence) if avg_confidence else 0.0,
                "profile_count": len(user_profiles_list)
            }
        
        # Comparar rasgos
        trait_comparisons = {}
        all_traits = set()
        for user_data in user_averages.values():
            all_traits.update(user_data["personality_traits"].keys())
        
        for trait in all_traits:
            trait_values = {
                str(user_id): user_data["personality_traits"].get(trait, 0.0)
                for user_id, user_data in user_averages.items()
            }
            
            if trait_values:
                max_user = max(trait_values, key=trait_values.get)
                min_user = min(trait_values, key=trait_values.get)
                avg_value = sum(trait_values.values()) / len(trait_values)
                
                trait_comparisons[trait] = {
                    "values": trait_values,
                    "average": avg_value,
                    "highest": {"user_id": max_user, "value": trait_values[max_user]},
                    "lowest": {"user_id": min_user, "value": trait_values[min_user]},
                    "range": trait_values[max_user] - trait_values[min_user]
                }
        
        # Comparar confianza
        confidence_values = {
            str(user_id): user_data["average_confidence"]
            for user_id, user_data in user_averages.items()
        }
        
        return {
            "users": {
                str(user_id): {
                    "average_confidence": user_data["average_confidence"],
                    "profile_count": user_data["profile_count"]
                }
                for user_id, user_data in user_averages.items()
            },
            "trait_comparisons": trait_comparisons,
            "confidence_comparison": {
                "values": confidence_values,
                "average": sum(confidence_values.values()) / len(confidence_values) if confidence_values else 0.0,
                "highest": {
                    "user_id": max(confidence_values, key=confidence_values.get),
                    "value": max(confidence_values.values())
                } if confidence_values else None,
                "lowest": {
                    "user_id": min(confidence_values, key=confidence_values.get),
                    "value": min(confidence_values.values())
                } if confidence_values else None
            },
            "total_users": len(user_averages)
        }
    
    def benchmark_against_population(
        self,
        profile: PsychologicalProfile,
        population_profiles: List[PsychologicalProfile]
    ) -> Dict[str, Any]:
        """
        Comparar perfil contra población
        
        Args:
            profile: Perfil a comparar
            population_profiles: Perfiles de población
            
        Returns:
            Benchmarking detallado
        """
        if not population_profiles:
            return {"error": "No population data available"}
        
        # Calcular promedios de población
        population_traits = defaultdict(list)
        population_confidence = []
        
        for pop_profile in population_profiles:
            for trait, value in pop_profile.personality_traits.items():
                population_traits[trait].append(value)
            population_confidence.append(pop_profile.confidence_score)
        
        # Comparar con población
        comparisons = {}
        for trait in profile.personality_traits:
            user_value = profile.personality_traits[trait]
            pop_values = population_traits.get(trait, [])
            
            if pop_values:
                pop_avg = sum(pop_values) / len(pop_values)
                pop_std = (
                    (sum((v - pop_avg)**2 for v in pop_values) / len(pop_values))**0.5
                    if len(pop_values) > 1 else 0.0
                )
                
                # Calcular percentil
                below_count = sum(1 for v in pop_values if v < user_value)
                percentile = (below_count / len(pop_values)) * 100
                
                comparisons[trait] = {
                    "user_value": user_value,
                    "population_average": pop_avg,
                    "population_std": pop_std,
                    "difference": user_value - pop_avg,
                    "percentile": percentile,
                    "interpretation": self._interpret_difference(user_value, pop_avg, pop_std)
                }
        
        # Comparar confianza
        user_confidence = profile.confidence_score
        pop_avg_confidence = sum(population_confidence) / len(population_confidence) if population_confidence else 0.0
        
        return {
            "trait_benchmarks": comparisons,
            "confidence_benchmark": {
                "user_value": user_confidence,
                "population_average": pop_avg_confidence,
                "difference": user_confidence - pop_avg_confidence,
                "percentile": (
                    (sum(1 for c in population_confidence if c < user_confidence) / len(population_confidence)) * 100
                    if population_confidence else 0.0
                )
            },
            "population_size": len(population_profiles)
        }
    
    def _interpret_difference(
        self,
        user_value: float,
        pop_avg: float,
        pop_std: float
    ) -> str:
        """Interpretar diferencia con población"""
        diff = user_value - pop_avg
        
        if pop_std == 0:
            if diff > 0:
                return "above_average"
            elif diff < 0:
                return "below_average"
            else:
                return "average"
        
        z_score = diff / pop_std if pop_std > 0 else 0
        
        if z_score > 2:
            return "significantly_above"
        elif z_score > 1:
            return "above_average"
        elif z_score < -2:
            return "significantly_below"
        elif z_score < -1:
            return "below_average"
        else:
            return "average"


# Instancia global del analizador comparativo
comparative_analyzer = ComparativeAnalyzer()




