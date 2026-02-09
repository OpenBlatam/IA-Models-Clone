"""
Comprehensive Unit Tests for Utility Functions
Tests for data_utils, helpers, and other utility functions
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Dict

# Import utilities
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


class TestDataUtils:
    """Test suite for data utility functions"""
    
    def test_normalize_features_basic(self):
        """Test basic feature normalization"""
        features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        normalized, mean, std = normalize_features(features)
        
        assert normalized.shape == features.shape
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-10)
    
    def test_normalize_features_with_mean_std(self):
        """Test normalization with provided mean and std"""
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean = np.array([2.0, 3.0])
        std = np.array([1.0, 1.0])
        normalized, returned_mean, returned_std = normalize_features(features, mean, std)
        
        assert np.allclose(returned_mean, mean)
        assert np.allclose(returned_std, std)
        assert np.allclose(normalized, (features - mean) / std)
    
    def test_normalize_features_zero_std(self):
        """Test normalization with zero standard deviation"""
        features = np.array([[1.0, 2.0], [1.0, 2.0]])
        normalized, mean, std = normalize_features(features)
        
        # Should handle zero std gracefully
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))
    
    def test_split_data_basic(self):
        """Test basic data splitting"""
        data = list(range(100))
        train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train) + len(val) + len(test) == 100
    
    def test_split_data_no_shuffle(self):
        """Test data splitting without shuffle"""
        data = list(range(100))
        train, val, test = split_data(data, shuffle=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert train == list(range(70))
        assert val == list(range(70, 85))
        assert test == list(range(85, 100))
    
    def test_split_data_with_seed(self):
        """Test data splitting with seed for reproducibility"""
        data = list(range(100))
        train1, val1, test1 = split_data(data, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        train2, val2, test2 = split_data(data, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2
    
    def test_split_data_invalid_ratios(self):
        """Test data splitting with invalid ratios"""
        data = list(range(100))
        with pytest.raises(ValueError):
            split_data(data, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    
    def test_split_data_empty(self):
        """Test data splitting with empty data"""
        data = []
        train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0
    
    def test_create_sequences_basic(self):
        """Test basic sequence creation"""
        data = [
            {"feature1": 1.0, "feature2": 2.0, "target": 0.5},
            {"feature1": 2.0, "feature2": 3.0, "target": 0.6},
            {"feature1": 3.0, "feature2": 4.0, "target": 0.7},
            {"feature1": 4.0, "feature2": 5.0, "target": 0.8}
        ]
        sequences, targets = create_sequences(data, sequence_length=2, feature_keys=["feature1", "feature2"], target_key="target")
        
        assert len(sequences) == 3
        assert len(targets) == 3
        assert sequences[0].shape == (2, 2)
        assert targets[0] == 0.6
    
    def test_create_sequences_no_target(self):
        """Test sequence creation without target"""
        data = [
            {"feature1": 1.0, "feature2": 2.0},
            {"feature1": 2.0, "feature2": 3.0},
            {"feature1": 3.0, "feature2": 4.0}
        ]
        sequences, targets = create_sequences(data, sequence_length=2, feature_keys=["feature1", "feature2"])
        
        assert len(sequences) == 2
        assert targets is None
    
    def test_create_sequences_short_data(self):
        """Test sequence creation with data shorter than sequence length"""
        data = [{"feature1": 1.0}]
        sequences, targets = create_sequences(data, sequence_length=3, feature_keys=["feature1"])
        
        assert len(sequences) == 0
    
    def test_create_sequences_missing_keys(self):
        """Test sequence creation with missing feature keys"""
        data = [
            {"feature1": 1.0},
            {"feature1": 2.0},
            {"feature1": 3.0}
        ]
        sequences, targets = create_sequences(data, sequence_length=2, feature_keys=["feature1", "feature2"])
        
        # Should use 0.0 for missing keys
        assert len(sequences) == 2
        assert sequences[0].shape == (2, 2)
    
    def test_augment_data_basic(self):
        """Test basic data augmentation"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        augmented = augment_data(data, noise_level=0.01, num_augmentations=2)
        
        assert augmented.shape[0] == 3  # Original + 2 augmentations
        assert augmented.shape[1:] == data.shape[1:]
        assert np.allclose(augmented[0], data)
    
    def test_augment_data_zero_noise(self):
        """Test augmentation with zero noise"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        augmented = augment_data(data, noise_level=0.0, num_augmentations=1)
        
        assert augmented.shape[0] == 2
        assert np.allclose(augmented[0], data)
    
    def test_augment_data_no_augmentations(self):
        """Test augmentation with zero augmentations"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        augmented = augment_data(data, num_augmentations=0)
        
        assert augmented.shape == data.shape
        assert np.allclose(augmented, data)
    
    def test_balance_dataset_undersample(self):
        """Test dataset balancing with undersampling"""
        data = [
            {"label": "A", "feature": 1},
            {"label": "A", "feature": 2},
            {"label": "A", "feature": 3},
            {"label": "B", "feature": 4},
            {"label": "B", "feature": 5}
        ]
        balanced = balance_dataset(data, target_key="label", method="undersample")
        
        # Should have equal counts for each class
        class_counts = {}
        for item in balanced:
            label = item["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        assert len(set(class_counts.values())) == 1  # All classes have same count
    
    def test_balance_dataset_oversample(self):
        """Test dataset balancing with oversampling"""
        data = [
            {"label": "A", "feature": 1},
            {"label": "A", "feature": 2},
            {"label": "B", "feature": 3}
        ]
        balanced = balance_dataset(data, target_key="label", method="oversample")
        
        # Should have equal counts for each class
        class_counts = {}
        for item in balanced:
            label = item["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        assert len(set(class_counts.values())) == 1
    
    def test_balance_dataset_single_class(self):
        """Test balancing with single class"""
        data = [
            {"label": "A", "feature": 1},
            {"label": "A", "feature": 2}
        ]
        balanced = balance_dataset(data, target_key="label")
        
        assert len(balanced) == len(data)
    
    def test_create_cross_validation_splits_basic(self):
        """Test basic cross-validation splits"""
        data = list(range(100))
        splits = create_cross_validation_splits(data, n_splits=5)
        
        assert len(splits) == 5
        for train, val in splits:
            assert len(train) + len(val) == 100
            assert len(set(train) & set(val)) == 0  # No overlap
    
    def test_create_cross_validation_splits_with_seed(self):
        """Test cross-validation splits with seed"""
        data = list(range(100))
        splits1 = create_cross_validation_splits(data, n_splits=5, seed=42)
        splits2 = create_cross_validation_splits(data, n_splits=5, seed=42)
        
        for (train1, val1), (train2, val2) in zip(splits1, splits2):
            assert train1 == train2
            assert val1 == val2
    
    def test_create_cross_validation_splits_no_shuffle(self):
        """Test cross-validation splits without shuffle"""
        data = list(range(20))
        splits = create_cross_validation_splits(data, n_splits=4, shuffle=False)
        
        # First split should have first 5 as validation
        train, val = splits[0]
        assert val == list(range(5))
        assert train == list(range(5, 20))


class TestHelpers:
    """Test suite for helper functions"""
    
    def test_calculate_days_sober_with_date(self):
        """Test calculating days sober with date"""
        last_use = datetime.now() - timedelta(days=10)
        days = calculate_days_sober(last_use)
        assert days == 10
    
    def test_calculate_days_sober_none(self):
        """Test calculating days sober with None"""
        days = calculate_days_sober(None)
        assert days == 0
    
    def test_calculate_days_sober_future_date(self):
        """Test calculating days sober with future date"""
        future_date = datetime.now() + timedelta(days=5)
        days = calculate_days_sober(future_date)
        assert days == 0  # Should not be negative
    
    def test_calculate_days_sober_with_current_date(self):
        """Test calculating days sober with specified current date"""
        last_use = datetime(2024, 1, 1)
        current = datetime(2024, 1, 11)
        days = calculate_days_sober(last_use, current)
        assert days == 10
    
    def test_calculate_money_saved_basic(self):
        """Test calculating money saved"""
        saved = calculate_money_saved(30, 10.0)
        assert saved == 300.0
    
    def test_calculate_money_saved_zero_days(self):
        """Test calculating money saved with zero days"""
        saved = calculate_money_saved(0, 10.0)
        assert saved == 0.0
    
    def test_calculate_money_saved_zero_cost(self):
        """Test calculating money saved with zero cost"""
        saved = calculate_money_saved(30, 0.0)
        assert saved == 0.0
    
    def test_calculate_health_improvements_cigarettes(self):
        """Test health improvements for cigarettes"""
        improvements = calculate_health_improvements(30, "cigarrillos")
        
        assert "circulation" in improvements
        assert "lung_capacity" in improvements
        assert "heart_rate" in improvements
        assert improvements["circulation"] > 0
        assert improvements["lung_capacity"] > 0
    
    def test_calculate_health_improvements_alcohol(self):
        """Test health improvements for alcohol"""
        improvements = calculate_health_improvements(30, "alcohol")
        
        assert "liver_function" in improvements
        assert "sleep_quality" in improvements
        assert "energy_level" in improvements
        assert improvements["sleep_quality"] > 0
    
    def test_calculate_health_improvements_capped(self):
        """Test health improvements are capped at 100"""
        improvements = calculate_health_improvements(1000, "cigarrillos")
        
        assert improvements["circulation"] <= 100
        assert improvements["lung_capacity"] <= 100
    
    def test_get_milestone_message_day_one(self):
        """Test milestone message for day 1"""
        message = get_milestone_message(1)
        assert message is not None
        assert "primer día" in message.lower() or "first day" in message.lower()
    
    def test_get_milestone_message_week(self):
        """Test milestone message for week"""
        message = get_milestone_message(7)
        assert message is not None
        assert "semana" in message.lower() or "week" in message.lower()
    
    def test_get_milestone_message_month(self):
        """Test milestone message for month"""
        message = get_milestone_message(30)
        assert message is not None
        assert "mes" in message.lower() or "month" in message.lower()
    
    def test_get_milestone_message_year(self):
        """Test milestone message for year"""
        message = get_milestone_message(365)
        assert message is not None
        assert "año" in message.lower() or "year" in message.lower()
    
    def test_get_milestone_message_no_milestone(self):
        """Test milestone message for non-milestone day"""
        message = get_milestone_message(15)
        assert message is None
    
    def test_calculate_relapse_risk_score_low(self):
        """Test relapse risk score calculation for low risk"""
        score = calculate_relapse_risk_score(
            days_sober=100,
            stress_level=2,
            support_level=9,
            triggers_present=False,
            previous_relapses=0
        )
        assert 0 <= score <= 100
        assert score < 50  # Should be low risk
    
    def test_calculate_relapse_risk_score_high(self):
        """Test relapse risk score calculation for high risk"""
        score = calculate_relapse_risk_score(
            days_sober=5,
            stress_level=9,
            support_level=2,
            triggers_present=True,
            previous_relapses=3
        )
        assert 0 <= score <= 100
        assert score > 50  # Should be high risk
    
    def test_calculate_relapse_risk_score_bounds(self):
        """Test relapse risk score stays within bounds"""
        # Test maximum risk
        score = calculate_relapse_risk_score(
            days_sober=1,
            stress_level=10,
            support_level=1,
            triggers_present=True,
            previous_relapses=10
        )
        assert score <= 100
        
        # Test minimum risk
        score = calculate_relapse_risk_score(
            days_sober=1000,
            stress_level=1,
            support_level=10,
            triggers_present=False,
            previous_relapses=0
        )
        assert score >= 0
    
    def test_calculate_relapse_risk_score_early_days(self):
        """Test relapse risk for early days of sobriety"""
        score1 = calculate_relapse_risk_score(3, 5, 5, False, 0)
        score2 = calculate_relapse_risk_score(50, 5, 5, False, 0)
        
        assert score1 > score2  # Early days should have higher risk
    
    def test_calculate_relapse_risk_score_stress_impact(self):
        """Test stress level impact on risk score"""
        score_low = calculate_relapse_risk_score(30, 2, 5, False, 0)
        score_high = calculate_relapse_risk_score(30, 9, 5, False, 0)
        
        assert score_high > score_low
    
    def test_format_time_sober_days(self):
        """Test formatting time sober in days"""
        formatted = format_time_sober(5)
        assert "5" in formatted
        assert "día" in formatted.lower() or "day" in formatted.lower()
    
    def test_format_time_sober_weeks(self):
        """Test formatting time sober in weeks"""
        formatted = format_time_sober(14)
        assert "2" in formatted or "semana" in formatted.lower() or "week" in formatted.lower()
    
    def test_format_time_sober_months(self):
        """Test formatting time sober in months"""
        formatted = format_time_sober(60)
        assert "mes" in formatted.lower() or "month" in formatted.lower()
    
    def test_format_time_sober_years(self):
        """Test formatting time sober in years"""
        formatted = format_time_sober(365)
        assert "año" in formatted.lower() or "year" in formatted.lower()
    
    def test_format_time_sober_zero(self):
        """Test formatting zero days"""
        formatted = format_time_sober(0)
        assert len(formatted) > 0


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def test_normalize_features_single_sample(self):
        """Test normalization with single sample"""
        features = np.array([[1.0, 2.0, 3.0]])
        normalized, mean, std = normalize_features(features)
        
        assert normalized.shape == features.shape
        assert not np.any(np.isnan(normalized))
    
    def test_normalize_features_single_feature(self):
        """Test normalization with single feature"""
        features = np.array([[1.0], [2.0], [3.0]])
        normalized, mean, std = normalize_features(features)
        
        assert normalized.shape == features.shape
        assert np.allclose(np.mean(normalized), 0, atol=1e-10)
    
    def test_split_data_small_dataset(self):
        """Test splitting very small dataset"""
        data = [1, 2, 3]
        train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert len(train) + len(val) + len(test) == 3
    
    def test_create_sequences_single_sequence(self):
        """Test creating sequences with data that creates only one sequence"""
        data = [
            {"feature1": 1.0},
            {"feature1": 2.0}
        ]
        sequences, targets = create_sequences(data, sequence_length=2, feature_keys=["feature1"])
        
        assert len(sequences) == 1
    
    def test_augment_data_single_sample(self):
        """Test augmentation with single sample"""
        data = np.array([[1.0, 2.0]])
        augmented = augment_data(data, num_augmentations=1)
        
        assert augmented.shape[0] == 2
    
    def test_balance_dataset_empty(self):
        """Test balancing empty dataset"""
        data = []
        balanced = balance_dataset(data, target_key="label")
        
        assert len(balanced) == 0
    
    def test_calculate_days_sober_same_day(self):
        """Test calculating days sober for same day"""
        now = datetime.now()
        days = calculate_days_sober(now, now)
        assert days == 0
    
    def test_calculate_health_improvements_zero_days(self):
        """Test health improvements with zero days"""
        improvements = calculate_health_improvements(0, "cigarrillos")
        
        assert improvements["circulation"] == 0
        assert improvements["lung_capacity"] == 0
    
    def test_calculate_relapse_risk_score_extreme_values(self):
        """Test relapse risk with extreme values"""
        # All factors at maximum
        score = calculate_relapse_risk_score(0, 10, 1, True, 100)
        assert 0 <= score <= 100
        
        # All factors at minimum
        score = calculate_relapse_risk_score(10000, 1, 10, False, 0)
        assert 0 <= score <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


