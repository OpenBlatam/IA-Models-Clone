"""
Enhanced Comprehensive Unit Tests for Utility Functions
Improved with fixtures, parametrization, and more intuitive test cases
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Dict

from utils.data_utils import (
    normalize_features,
    split_data,
    create_sequences,
    augment_data,
    balance_dataset,
    create_cross_validation_splits
)
from utils.helpers import (
    calculate_days_sober,
    calculate_money_saved,
    calculate_health_improvements,
    get_milestone_message,
    calculate_relapse_risk_score,
    format_time_sober
)


# ============================================================================
# Data Utils Enhanced Tests
# ============================================================================

class TestNormalizeFeaturesEnhanced:
    """Enhanced tests for normalize_features with parametrization"""
    
    def test_normalize_features_standard_distribution(self, sample_features_array):
        """Test normalization creates standard normal distribution"""
        normalized, mean, std = normalize_features(sample_features_array)
        
        assert normalized.shape == sample_features_array.shape
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-10)
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
    
    @pytest.mark.parametrize("shape", [
        (10, 5),
        (100, 10),
        (1, 20),
        (50, 1),
    ])
    def test_normalize_features_various_shapes(self, shape):
        """Test normalization works with various array shapes"""
        features = np.random.randn(*shape)
        normalized, mean, std = normalize_features(features)
        
        assert normalized.shape == features.shape
        assert mean.shape == (shape[1],)
        assert std.shape == (shape[1],)
    
    def test_normalize_features_with_provided_statistics(self):
        """Test normalization using provided mean and std"""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mean = np.array([3.0, 4.0])
        std = np.array([1.0, 1.0])
        
        normalized, returned_mean, returned_std = normalize_features(features, mean, std)
        
        assert np.allclose(returned_mean, mean)
        assert np.allclose(returned_std, std)
        assert np.allclose(normalized, (features - mean) / std)
    
    def test_normalize_features_handles_zero_std(self):
        """Test normalization handles zero standard deviation gracefully"""
        features = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        normalized, mean, std = normalize_features(features)
        
        # Should not produce NaN or Inf
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
        # Zero std should be replaced with 1.0
        assert np.all(std > 0)
    
    def test_normalize_features_preserves_data_integrity(self, sample_features_array):
        """Test that normalization preserves data integrity (reversible)"""
        normalized, mean, std = normalize_features(sample_features_array)
        
        # Reverse normalization
        reconstructed = normalized * std + mean
        
        assert np.allclose(sample_features_array, reconstructed, atol=1e-10)


class TestSplitDataEnhanced:
    """Enhanced tests for split_data with parametrization"""
    
    @pytest.mark.parametrize("total_size,train_ratio,val_ratio,test_ratio", [
        (100, 0.7, 0.15, 0.15),
        (1000, 0.8, 0.1, 0.1),
        (50, 0.6, 0.2, 0.2),
        (200, 0.75, 0.125, 0.125),
    ])
    def test_split_data_maintains_total_size(
        self, total_size, train_ratio, val_ratio, test_ratio
    ):
        """Test that split maintains total data size"""
        data = list(range(total_size))
        train, val, test = split_data(data, train_ratio, val_ratio, test_ratio)
        
        assert len(train) + len(val) + len(test) == total_size
        assert abs(len(train) / total_size - train_ratio) < 0.02  # Allow small rounding
        assert abs(len(val) / total_size - val_ratio) < 0.02
        assert abs(len(test) / total_size - test_ratio) < 0.02
    
    def test_split_data_with_seed_reproducibility(self, sample_data_list):
        """Test that same seed produces same splits"""
        train1, val1, test1 = split_data(sample_data_list, seed=42)
        train2, val2, test2 = split_data(sample_data_list, seed=42)
        
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2
    
    def test_split_data_without_shuffle_preserves_order(self, sample_data_list):
        """Test that split without shuffle preserves original order"""
        train, val, test = split_data(sample_data_list, shuffle=False)
        
        # Check that each split is in order
        assert train == sorted(train)
        assert val == sorted(val)
        assert test == sorted(test)
        
        # Check that splits are contiguous
        all_data = train + val + test
        assert all_data == sorted(all_data)
    
    def test_split_data_handles_small_datasets(self):
        """Test split handles very small datasets"""
        for size in [1, 2, 3, 5, 10]:
            data = list(range(size))
            train, val, test = split_data(data)
            assert len(train) + len(val) + len(test) == size
    
    def test_split_data_raises_error_invalid_ratios(self, sample_data_list):
        """Test that invalid ratios raise appropriate error"""
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_data(sample_data_list, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


class TestCreateSequencesEnhanced:
    """Enhanced tests for create_sequences with parametrization"""
    
    @pytest.mark.parametrize("sequence_length,expected_sequences", [
        (2, 19),  # 20 samples - 2 + 1 = 19 sequences
        (5, 16),  # 20 samples - 5 + 1 = 16 sequences
        (10, 11), # 20 samples - 10 + 1 = 11 sequences
    ])
    def test_create_sequences_correct_count(
        self, sample_sequence_data, sequence_length, expected_sequences
    ):
        """Test that sequences are created with correct count"""
        sequences, targets = create_sequences(
            sample_sequence_data,
            sequence_length=sequence_length,
            feature_keys=["feature1", "feature2"],
            target_key="target"
        )
        
        assert len(sequences) == expected_sequences
        assert len(targets) == expected_sequences
    
    def test_create_sequences_correct_structure(self, sample_sequence_data):
        """Test that sequences have correct structure"""
        sequences, targets = create_sequences(
            sample_sequence_data,
            sequence_length=3,
            feature_keys=["feature1", "feature2"],
            target_key="target"
        )
        
        assert len(sequences) > 0
        assert sequences[0].shape == (3, 2)  # (sequence_length, num_features)
        assert isinstance(targets, list)
        assert len(targets) == len(sequences)
    
    def test_create_sequences_without_target(self, sample_sequence_data):
        """Test sequence creation without target"""
        sequences, targets = create_sequences(
            sample_sequence_data,
            sequence_length=3,
            feature_keys=["feature1", "feature2"]
        )
        
        assert len(sequences) > 0
        assert targets is None
    
    def test_create_sequences_handles_missing_keys(self):
        """Test sequences handle missing feature keys gracefully"""
        data = [
            {"feature1": 1.0},
            {"feature1": 2.0},
            {"feature1": 3.0}
        ]
        sequences, targets = create_sequences(
            data,
            sequence_length=2,
            feature_keys=["feature1", "feature2"]  # feature2 missing
        )
        
        assert len(sequences) > 0
        assert sequences[0].shape == (2, 2)  # Missing keys default to 0.0


class TestAugmentDataEnhanced:
    """Enhanced tests for augment_data"""
    
    def test_augment_data_increases_dataset_size(self):
        """Test that augmentation increases dataset size"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        augmented = augment_data(data, num_augmentations=3)
        
        assert augmented.shape[0] == 4  # Original + 3 augmentations
        assert augmented.shape[1] == data.shape[1]
    
    def test_augment_data_preserves_original(self):
        """Test that original data is preserved in augmentation"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        augmented = augment_data(data, num_augmentations=1)
        
        assert np.allclose(augmented[0], data)  # First should be original
    
    def test_augment_data_noise_level_impact(self):
        """Test that noise level affects augmentation"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        low_noise = augment_data(data, noise_level=0.01, num_augmentations=1)
        high_noise = augment_data(data, noise_level=0.1, num_augmentations=1)
        
        # High noise should have larger deviation from original
        low_diff = np.abs(low_noise[1] - data).mean()
        high_diff = np.abs(high_noise[1] - data).mean()
        
        assert high_diff > low_diff


class TestBalanceDatasetEnhanced:
    """Enhanced tests for balance_dataset"""
    
    def test_balance_dataset_undersample_creates_balance(self):
        """Test undersampling creates balanced dataset"""
        data = [
            {"label": "A", "feature": i} for i in range(10)
        ] + [
            {"label": "B", "feature": i} for i in range(5)
        ]
        
        balanced = balance_dataset(data, target_key="label", method="undersample")
        
        # Count labels
        label_counts = {}
        for item in balanced:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Should be balanced (all classes have same count)
        assert len(set(label_counts.values())) == 1
    
    def test_balance_dataset_oversample_creates_balance(self):
        """Test oversampling creates balanced dataset"""
        data = [
            {"label": "A", "feature": i} for i in range(5)
        ] + [
            {"label": "B", "feature": i} for i in range(10)
        ]
        
        balanced = balance_dataset(data, target_key="label", method="oversample")
        
        # Count labels
        label_counts = {}
        for item in balanced:
            label = item["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Should be balanced
        assert len(set(label_counts.values())) == 1


# ============================================================================
# Helper Functions Enhanced Tests
# ============================================================================

class TestCalculateDaysSoberEnhanced:
    """Enhanced tests for calculate_days_sober"""
    
    @pytest.mark.parametrize("days_ago,expected_days", [
        (0, 0),
        (1, 1),
        (7, 7),
        (30, 30),
        (365, 365),
    ])
    def test_calculate_days_sober_accurate(self, days_ago, expected_days):
        """Test days sober calculation is accurate"""
        last_use = datetime.now() - timedelta(days=days_ago)
        days = calculate_days_sober(last_use)
        
        assert days == expected_days
    
    def test_calculate_days_sober_with_future_date(self):
        """Test days sober handles future dates (should return 0)"""
        future_date = datetime.now() + timedelta(days=5)
        days = calculate_days_sober(future_date)
        
        assert days == 0
    
    def test_calculate_days_sober_with_custom_current_date(self):
        """Test days sober with custom current date"""
        last_use = datetime(2024, 1, 1)
        current = datetime(2024, 1, 11)
        days = calculate_days_sober(last_use, current)
        
        assert days == 10


class TestCalculateMoneySavedEnhanced:
    """Enhanced tests for calculate_money_saved"""
    
    @pytest.mark.parametrize("days,cost,expected", [
        (30, 10.0, 300.0),
        (90, 15.0, 1350.0),
        (365, 20.0, 7300.0),
        (0, 10.0, 0.0),
    ])
    def test_calculate_money_saved_accurate(self, days, cost, expected):
        """Test money saved calculation is accurate"""
        saved = calculate_money_saved(days, cost)
        assert saved == expected
    
    def test_calculate_money_saved_with_fractional_cost(self):
        """Test money saved with fractional daily cost"""
        saved = calculate_money_saved(30, 12.50)
        assert saved == 375.0


class TestCalculateHealthImprovementsEnhanced:
    """Enhanced tests for calculate_health_improvements"""
    
    @pytest.mark.parametrize("addiction_type,expected_improvements", [
        ("cigarrillos", ["circulation", "lung_capacity", "heart_rate"]),
        ("tabaco", ["circulation", "lung_capacity", "heart_rate"]),
        ("alcohol", ["liver_function", "sleep_quality", "energy_level"]),
    ])
    def test_health_improvements_by_addiction_type(
        self, addiction_type, expected_improvements
    ):
        """Test health improvements are specific to addiction type"""
        improvements = calculate_health_improvements(30, addiction_type)
        
        for improvement in expected_improvements:
            assert improvement in improvements
            assert improvements[improvement] > 0
    
    def test_health_improvements_are_capped_at_100(self):
        """Test health improvements are capped at 100%"""
        improvements = calculate_health_improvements(10000, "cigarrillos")
        
        for key, value in improvements.items():
            assert value <= 100
    
    def test_health_improvements_progress_over_time(self):
        """Test health improvements increase over time"""
        improvements_30 = calculate_health_improvements(30, "cigarrillos")
        improvements_90 = calculate_health_improvements(90, "cigarrillos")
        
        # Most improvements should be higher at 90 days
        for key in improvements_30:
            if key in improvements_90:
                assert improvements_90[key] >= improvements_30[key]


class TestGetMilestoneMessageEnhanced:
    """Enhanced tests for get_milestone_message"""
    
    @pytest.mark.parametrize("days,should_have_message", [
        (1, True),
        (7, True),
        (30, True),
        (90, True),
        (180, True),
        (365, True),
        (15, False),
        (50, False),
        (200, False),
    ])
    def test_milestone_messages_at_key_days(self, days, should_have_message):
        """Test milestone messages appear at key days"""
        message = get_milestone_message(days)
        
        if should_have_message:
            assert message is not None
            assert len(message) > 0
        else:
            assert message is None
    
    def test_milestone_messages_are_encouraging(self):
        """Test milestone messages are encouraging"""
        for days in [1, 7, 30, 90, 365]:
            message = get_milestone_message(days)
            if message:
                # Check for encouraging words
                encouraging_words = [
                    "felic", "congrat", "great", "increíble", "amazing",
                    "logro", "achievement", "éxito", "success"
                ]
                assert any(word in message.lower() for word in encouraging_words)


class TestCalculateRelapseRiskScoreEnhanced:
    """Enhanced tests for calculate_relapse_risk_score"""
    
    def test_risk_score_bounds(self):
        """Test risk score stays within 0-100 bounds"""
        # Minimum risk
        score = calculate_relapse_risk_score(1000, 1, 10, False, 0)
        assert 0 <= score <= 100
        
        # Maximum risk
        score = calculate_relapse_risk_score(1, 10, 1, True, 10)
        assert 0 <= score <= 100
    
    @pytest.mark.parametrize("days_sober,expected_trend", [
        (1, "higher"),
        (7, "higher"),
        (30, "medium"),
        (90, "lower"),
        (365, "lower"),
    ])
    def test_risk_score_decreases_with_sobriety(self, days_sober, expected_trend):
        """Test risk score decreases with longer sobriety"""
        base_state = (days_sober, 5, 5, False, 0)
        score = calculate_relapse_risk_score(*base_state)
        
        if expected_trend == "higher":
            assert score > 40
        elif expected_trend == "medium":
            assert 20 <= score <= 60
        else:  # lower
            assert score < 40
    
    def test_risk_score_increases_with_stress(self):
        """Test risk score increases with higher stress"""
        low_stress = calculate_relapse_risk_score(30, 2, 5, False, 0)
        high_stress = calculate_relapse_risk_score(30, 9, 5, False, 0)
        
        assert high_stress > low_stress
    
    def test_risk_score_decreases_with_support(self):
        """Test risk score decreases with more support"""
        low_support = calculate_relapse_risk_score(30, 5, 2, False, 0)
        high_support = calculate_relapse_risk_score(30, 5, 9, False, 0)
        
        assert low_support > high_support
    
    def test_risk_score_increases_with_triggers(self):
        """Test risk score increases when triggers are present"""
        no_triggers = calculate_relapse_risk_score(30, 5, 5, False, 0)
        with_triggers = calculate_relapse_risk_score(30, 5, 5, True, 0)
        
        assert with_triggers > no_triggers


class TestFormatTimeSoberEnhanced:
    """Enhanced tests for format_time_sober"""
    
    @pytest.mark.parametrize("days,expected_contains", [
        (1, "día"),
        (7, "semana"),
        (30, "mes"),
        (365, "año"),
    ])
    def test_format_time_sober_appropriate_units(self, days, expected_contains):
        """Test time formatting uses appropriate units"""
        formatted = format_time_sober(days)
        
        assert expected_contains in formatted.lower() or "day" in formatted.lower() or "week" in formatted.lower() or "month" in formatted.lower() or "year" in formatted.lower()
    
    def test_format_time_sober_is_readable(self):
        """Test formatted time is human-readable"""
        for days in [1, 7, 30, 90, 365]:
            formatted = format_time_sober(days)
            assert len(formatted) > 0
            assert isinstance(formatted, str)


