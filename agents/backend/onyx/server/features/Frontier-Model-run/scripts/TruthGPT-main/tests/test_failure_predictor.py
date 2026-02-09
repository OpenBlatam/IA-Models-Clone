"""
Test Failure Predictor
Uses machine learning to predict which tests are likely to fail
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class TestFailurePredictor:
    """Predict test failures using historical data and patterns"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.model_file = project_root / ".test_cache" / "failure_predictor_model.pkl"
        self.history_file = project_root / "test_failure_history.json"
        self.history = self._load_history()
        self.model = self._load_model()
    
    def _load_history(self) -> List[Dict]:
        """Load test failure history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    
    def _save_history(self):
        """Save test failure history"""
        self.history_file.parent.mkdir(exist_ok=True)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
    
    def _load_model(self) -> Optional[Dict]:
        """Load trained model"""
        if self.model_file.exists():
            try:
                with open(self.model_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_model(self, model: Dict):
        """Save trained model"""
        self.model_file.parent.mkdir(exist_ok=True)
        with open(self.model_file, 'wb') as f:
            pickle.dump(model, f)
    
    def record_test_result(
        self,
        test_name: str,
        status: str,
        duration: float,
        timestamp: datetime = None,
        error_type: str = None,
        code_changes: List[str] = None
    ):
        """Record a test result for training"""
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            'test_name': test_name,
            'status': status,
            'duration': duration,
            'timestamp': timestamp.isoformat(),
            'error_type': error_type,
            'code_changes': code_changes or [],
            'failed': status in ('failed', 'error')
        }
        
        self.history.append(record)
        
        # Keep only last 1000 records
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        self._save_history()
    
    def _extract_features(self, test_name: str, recent_history: List[Dict]) -> Dict:
        """Extract features for prediction"""
        test_records = [r for r in recent_history if r['test_name'] == test_name]
        
        if not test_records:
            return {
                'failure_rate': 0.0,
                'avg_duration': 0.0,
                'recent_failures': 0,
                'failure_trend': 0.0,
                'error_types_count': 0,
                'code_changes_correlation': 0.0
            }
        
        # Calculate failure rate
        failures = sum(1 for r in test_records if r['failed'])
        failure_rate = failures / len(test_records) if test_records else 0.0
        
        # Average duration
        durations = [r['duration'] for r in test_records if r.get('duration', 0) > 0]
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        # Recent failures (last 5 runs)
        recent = test_records[-5:]
        recent_failures = sum(1 for r in recent if r['failed'])
        
        # Failure trend (increasing/decreasing)
        if len(test_records) >= 2:
            first_half = test_records[:len(test_records)//2]
            second_half = test_records[len(test_records)//2:]
            first_rate = sum(1 for r in first_half if r['failed']) / len(first_half) if first_half else 0
            second_rate = sum(1 for r in second_half if r['failed']) / len(second_half) if second_half else 0
            failure_trend = second_rate - first_rate
        else:
            failure_trend = 0.0
        
        # Error types
        error_types = [r.get('error_type') for r in test_records if r.get('error_type')]
        error_types_count = len(set(error_types))
        
        # Code changes correlation
        changes_with_failures = sum(
            1 for r in test_records
            if r['failed'] and r.get('code_changes')
        )
        code_changes_correlation = changes_with_failures / len(test_records) if test_records else 0.0
        
        return {
            'failure_rate': failure_rate,
            'avg_duration': avg_duration,
            'recent_failures': recent_failures,
            'failure_trend': failure_trend,
            'error_types_count': error_types_count,
            'code_changes_correlation': code_changes_correlation
        }
    
    def train_model(self):
        """Train prediction model using historical data"""
        if len(self.history) < 10:
            print("⚠️ Not enough historical data for training (need at least 10 records)")
            return
        
        # Group by test name
        by_test = defaultdict(list)
        for record in self.history:
            by_test[record['test_name']].append(record)
        
        # Calculate features and outcomes
        model_data = {}
        for test_name, records in by_test.items():
            features = self._extract_features(test_name, records)
            # Outcome: did it fail in the most recent run?
            if records:
                latest = records[-1]
                outcome = latest['failed']
                model_data[test_name] = {
                    'features': features,
                    'outcome': outcome,
                    'confidence': min(len(records) / 20.0, 1.0)  # More data = more confidence
                }
        
        model = {
            'trained_at': datetime.now().isoformat(),
            'test_count': len(model_data),
            'data': model_data
        }
        
        self._save_model(model)
        self.model = model
        
        print(f"✅ Model trained on {len(model_data)} tests")
        return model
    
    def predict_failures(self, test_names: List[str] = None) -> List[Dict]:
        """Predict which tests are likely to fail"""
        if not self.model:
            print("⚠️ No trained model. Run train_model() first.")
            return []
        
        # Get recent history (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        recent_history = [
            r for r in self.history
            if datetime.fromisoformat(r['timestamp']) > cutoff
        ]
        
        predictions = []
        
        # If test_names provided, predict only those
        tests_to_predict = test_names if test_names else list(self.model['data'].keys())
        
        for test_name in tests_to_predict:
            if test_name not in self.model['data']:
                continue
            
            test_data = self.model['data'][test_name]
            features = test_data['features']
            confidence = test_data['confidence']
            
            # Calculate failure probability
            # Simple heuristic: combine features
            failure_prob = (
                features['failure_rate'] * 0.4 +
                (features['recent_failures'] / 5.0) * 0.3 +
                max(0, features['failure_trend']) * 0.2 +
                features['code_changes_correlation'] * 0.1
            )
            
            # Adjust based on historical outcome
            if test_data['outcome']:
                failure_prob += 0.2
            
            failure_prob = min(failure_prob, 1.0)
            
            predictions.append({
                'test_name': test_name,
                'failure_probability': round(failure_prob, 3),
                'confidence': round(confidence, 3),
                'risk_level': self._get_risk_level(failure_prob),
                'features': features,
                'recommendation': self._get_recommendation(failure_prob, features)
            })
        
        # Sort by failure probability
        predictions.sort(key=lambda x: x['failure_probability'], reverse=True)
        
        return predictions
    
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level from probability"""
        if probability >= 0.7:
            return 'high'
        elif probability >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommendation(self, probability: float, features: Dict) -> str:
        """Get recommendation based on prediction"""
        if probability >= 0.7:
            return "Run this test first and monitor closely"
        elif probability >= 0.4:
            return "Consider running this test early in the suite"
        elif features['failure_rate'] > 0.5:
            return "This test has high historical failure rate"
        else:
            return "Low risk, can run later"
    
    def get_prioritized_test_order(self, test_names: List[str]) -> List[str]:
        """Get recommended test execution order based on predictions"""
        predictions = self.predict_failures(test_names)
        
        # Sort by failure probability (highest first)
        prioritized = [p['test_name'] for p in predictions]
        
        # Add any tests not in predictions
        for test_name in test_names:
            if test_name not in prioritized:
                prioritized.append(test_name)
        
        return prioritized


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Failure Predictor')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', nargs='*', help='Predict failures for specific tests')
    parser.add_argument('--prioritize', nargs='+', help='Get prioritized order for tests')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    predictor = TestFailurePredictor(project_root)
    
    if args.train:
        print("🔮 Training failure prediction model...")
        predictor.train_model()
    elif args.predict is not None:
        print("🔮 Predicting test failures...")
        predictions = predictor.predict_failures(args.predict if args.predict else None)
        
        print(f"\n📊 Predictions ({len(predictions)} tests):")
        for pred in predictions[:20]:  # Show top 20
            print(f"\n  {pred['test_name']}")
            print(f"    Risk: {pred['risk_level'].upper()} ({pred['failure_probability']*100:.1f}% failure probability)")
            print(f"    Confidence: {pred['confidence']*100:.1f}%")
            print(f"    Recommendation: {pred['recommendation']}")
    elif args.prioritize:
        print("🔮 Prioritizing test execution order...")
        order = predictor.get_prioritized_test_order(args.prioritize)
        print("\n📋 Recommended execution order:")
        for i, test_name in enumerate(order, 1):
            print(f"  {i}. {test_name}")
    else:
        print("Use --help to see available options")


if __name__ == '__main__':
    main()

