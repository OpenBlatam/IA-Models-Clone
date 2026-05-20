"""
Gamification domain services
"""

from services.domains import register_service

try:
    from services.gamification_service import GamificationService
    from services.advanced_gamification_service import AdvancedGamificationService
    from services.advanced_achievements_service import AdvancedAchievementsService
    from services.advanced_rewards_service import AdvancedRewardsService
    from services.virtual_economy_service import VirtualEconomyService
    
    def register_services():
        register_service("gamification", "gamification", GamificationService)
        register_service("gamification", "advanced", AdvancedGamificationService)
        register_service("gamification", "achievements", AdvancedAchievementsService)
        register_service("gamification", "rewards", AdvancedRewardsService)
        register_service("gamification", "virtual_economy", VirtualEconomyService)
except ImportError:
    pass



