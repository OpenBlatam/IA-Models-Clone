"""
Tests modulares para RankingService
"""

import pytest
from datetime import datetime, timedelta
from services import RankingService
from exceptions import ValueError, TypeError


class TestRankingService:
    """Tests para RankingService"""
    
    @pytest.fixture
    def ranking_service(self):
        """Fixture para RankingService"""
        return RankingService()
    
    @pytest.mark.unit
    def test_calculate_score_basic(self, ranking_service):
        """Test de cálculo básico de score"""
        score = ranking_service.calculate_score(
            vote_count=10,
            remix_count=5,
            view_count=100,
            created_at=datetime.utcnow()
        )
        
        assert isinstance(score, float)
        assert score >= 0
        assert score == round(score, 2)  # Debe estar redondeado
    
    @pytest.mark.unit
    def test_calculate_score_zero_engagement(self, ranking_service):
        """Test con engagement cero"""
        score = ranking_service.calculate_score(
            vote_count=0,
            remix_count=0,
            view_count=0,
            created_at=datetime.utcnow()
        )
        
        assert score == 0.0
    
    @pytest.mark.unit
    def test_calculate_score_high_engagement(self, ranking_service):
        """Test con alto engagement"""
        score = ranking_service.calculate_score(
            vote_count=1000,
            remix_count=500,
            view_count=10000,
            created_at=datetime.utcnow()
        )
        
        assert score > 0
        assert isinstance(score, float)
    
    @pytest.mark.unit
    def test_calculate_score_time_decay(self, ranking_service):
        """Test que el score decae con el tiempo"""
        now = datetime.utcnow()
        recent = now - timedelta(hours=1)
        old = now - timedelta(days=7)
        
        score_recent = ranking_service.calculate_score(10, 5, 100, recent)
        score_old = ranking_service.calculate_score(10, 5, 100, old)
        
        # El score reciente debe ser mayor
        assert score_recent > score_old, f"Recent score {score_recent} should be > old score {score_old}"
    
    @pytest.mark.unit
    def test_calculate_score_with_base_score(self, ranking_service):
        """Test con score base adicional"""
        score_with_base = ranking_service.calculate_score(
            vote_count=10,
            remix_count=5,
            view_count=100,
            created_at=datetime.utcnow(),
            base_score=5.0
        )
        
        score_without_base = ranking_service.calculate_score(
            vote_count=10,
            remix_count=5,
            view_count=100,
            created_at=datetime.utcnow(),
            base_score=0.0
        )
        
        assert score_with_base > score_without_base
    
    @pytest.mark.error_handling
    def test_calculate_score_negative_vote_count(self, ranking_service):
        """Test con vote_count negativo"""
        with pytest.raises(ValueError):
            ranking_service.calculate_score(
                vote_count=-1,
                remix_count=0,
                view_count=0,
                created_at=datetime.utcnow()
            )
    
    @pytest.mark.error_handling
    def test_calculate_score_negative_remix_count(self, ranking_service):
        """Test con remix_count negativo"""
        with pytest.raises(ValueError):
            ranking_service.calculate_score(
                vote_count=0,
                remix_count=-1,
                view_count=0,
                created_at=datetime.utcnow()
            )
    
    @pytest.mark.error_handling
    def test_calculate_score_negative_view_count(self, ranking_service):
        """Test con view_count negativo"""
        with pytest.raises(ValueError):
            ranking_service.calculate_score(
                vote_count=0,
                remix_count=0,
                view_count=-1,
                created_at=datetime.utcnow()
            )
    
    @pytest.mark.error_handling
    def test_calculate_score_invalid_datetime(self, ranking_service):
        """Test con datetime inválido"""
        with pytest.raises(TypeError):
            ranking_service.calculate_score(
                vote_count=10,
                remix_count=5,
                view_count=100,
                created_at="not-a-datetime"
            )
    
    @pytest.mark.edge_case
    def test_calculate_score_very_recent(self, ranking_service):
        """Test con contenido muy reciente (sin decay)"""
        very_recent = datetime.utcnow() - timedelta(minutes=1)
        
        score = ranking_service.calculate_score(
            vote_count=10,
            remix_count=5,
            view_count=100,
            created_at=very_recent
        )
        
        assert score > 0
    
    @pytest.mark.edge_case
    def test_calculate_score_very_old(self, ranking_service):
        """Test con contenido muy antiguo"""
        very_old = datetime.utcnow() - timedelta(days=365)
        
        score = ranking_service.calculate_score(
            vote_count=1000,
            remix_count=500,
            view_count=10000,
            created_at=very_old
        )
        
        # Debe tener score pero menor debido al decay
        assert score > 0
        assert score < 1000  # Debe estar decayed
    
    @pytest.mark.edge_case
    def test_calculate_score_extreme_values(self, ranking_service):
        """Test con valores extremos"""
        score = ranking_service.calculate_score(
            vote_count=1000000,
            remix_count=500000,
            view_count=10000000,
            created_at=datetime.utcnow()
        )
        
        assert score > 0
        assert isinstance(score, float)

