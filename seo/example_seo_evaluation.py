#!/usr/bin/env python3
"""
SEO Evaluation Metrics Example
Demonstrates how to use the SEO evaluation system with your existing models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import json
from pathlib import Path

# Import the evaluation system
from evaluation_metrics import (
    SEOModelEvaluator, SEOMetricsConfig, ClassificationMetricsConfig, 
    RegressionMetricsConfig, create_seo_test_data
)

# =============================================================================
# SEO MODEL EVALUATION EXAMPLE
# =============================================================================

class SEOKeywordClassifier(nn.Module):
    """Example SEO keyword classification model."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Model architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

class SEORankingModel(nn.Module):
    """Example SEO ranking model."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.feature_layers(x)
        ranking_score = self.ranking_head(features)
        return ranking_score

class SEOModelEvaluatorExample:
    """Example of how to use the SEO evaluation system."""
    
    def __init__(self):
        self.evaluator = None
        self.results = {}
    
    async def setup_evaluation(self):
        """Setup the evaluation system."""
        print("🔧 Setting up SEO evaluation system...")
        
        # SEO-specific configuration
        seo_config = SEOMetricsConfig(
            ranking_metrics=True,
            content_quality_metrics=True,
            user_engagement_metrics=True,
            technical_seo_metrics=True,
            ndcg_k_values=[1, 3, 5, 10],
            map_k_values=[1, 3, 5, 10]
        )
        
        # Classification metrics configuration
        classification_config = ClassificationMetricsConfig(
            average="weighted",
            zero_division=0
        )
        
        # Regression metrics configuration
        regression_config = RegressionMetricsConfig(
            multioutput="uniform_average"
        )
        
        # Create evaluator
        self.evaluator = SEOModelEvaluator(
            seo_config=seo_config,
            classification_config=classification_config,
            regression_config=regression_config
        )
        
        print("✅ Evaluation system setup completed")
    
    async def evaluate_keyword_classification(self, model: nn.Module, test_data: Dict[str, Any]):
        """Evaluate keyword classification model."""
        print("\n📊 Evaluating Keyword Classification Model...")
        
        try:
            results = await self.evaluator.evaluate_seo_model(
                model, test_data, task_type="classification"
            )
            
            # Store results
            self.results['keyword_classification'] = results
            
            # Print key metrics
            print(f"  ✅ Classification Accuracy: {results.get('accuracy', 0):.4f}")
            print(f"  ✅ F1 Score: {results.get('f1_score', 0):.4f}")
            print(f"  ✅ Precision: {results.get('precision', 0):.4f}")
            print(f"  ✅ Recall: {results.get('recall', 0):.4f}")
            
            if 'roc_auc' in results:
                print(f"  ✅ ROC AUC: {results.get('roc_auc', 0):.4f}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Keyword classification evaluation failed: {e}")
            return None
    
    async def evaluate_content_ranking(self, model: nn.Module, test_data: Dict[str, Any]):
        """Evaluate content ranking model."""
        print("\n📊 Evaluating Content Ranking Model...")
        
        try:
            results = await self.evaluator.evaluate_seo_model(
                model, test_data, task_type="ranking"
            )
            
            # Store results
            self.results['content_ranking'] = results
            
            # Print key metrics
            print(f"  ✅ NDCG@5: {results.get('ndcg_at_5', 0):.4f}")
            print(f"  ✅ MAP@5: {results.get('map_at_5', 0):.4f}")
            print(f"  ✅ MRR: {results.get('mrr', 0):.4f}")
            print(f"  ✅ Content Quality Score: {results.get('overall_content_quality', 0):.4f}")
            print(f"  ✅ User Engagement Score: {results.get('overall_engagement_score', 0):.4f}")
            print(f"  ✅ Technical SEO Score: {results.get('overall_technical_score', 0):.4f}")
            
            return results
            
        except Exception as e:
            print(f"  ❌ Content ranking evaluation failed: {e}")
            return None
    
    async def evaluate_technical_seo(self, test_data: Dict[str, Any]):
        """Evaluate technical SEO metrics."""
        print("\n📊 Evaluating Technical SEO Metrics...")
        
        try:
            # Extract technical data
            technical_data = test_data.get('technical_data', {})
            
            if not technical_data:
                print("  ⚠️ No technical SEO data available")
                return None
            
            # Calculate technical metrics
            technical_metrics = self.evaluator.seo_metrics.calculate_technical_seo_metrics(
                technical_data
            )
            
            # Store results
            self.results['technical_seo'] = technical_metrics
            
            # Print key metrics
            print(f"  ✅ Page Load Speed Score: {technical_metrics.get('load_speed_score', 0):.4f}")
            print(f"  ✅ Mobile Friendliness: {technical_metrics.get('mobile_score_normalized', 0):.4f}")
            print(f"  ✅ LCP Score: {technical_metrics.get('lcp_score', 0):.4f}")
            print(f"  ✅ FID Score: {technical_metrics.get('fid_score', 0):.4f}")
            print(f"  ✅ CLS Score: {technical_metrics.get('cls_score', 0):.4f}")
            print(f"  ✅ Overall Technical Score: {technical_metrics.get('overall_technical_score', 0):.4f}")
            
            return technical_metrics
            
        except Exception as e:
            print(f"  ❌ Technical SEO evaluation failed: {e}")
            return None
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.evaluator:
            return {}
        
        # Get evaluation summary
        summary = self.evaluator.get_evaluation_summary()
        
        # Add custom analysis
        report = {
            'evaluation_summary': summary,
            'model_performance': self._analyze_model_performance(),
            'seo_insights': self._generate_seo_insights(),
            'improvement_recommendations': summary.get('recommendations', []),
            'next_steps': self._suggest_next_steps()
        }
        
        return report
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze overall model performance."""
        analysis = {
            'overall_score': 0.0,
            'task_performance': {},
            'strengths': [],
            'weaknesses': []
        }
        
        if not self.results:
            return analysis
        
        # Calculate overall score
        scores = []
        for task, results in self.results.items():
            if task == 'keyword_classification':
                score = results.get('f1_score', 0.0)
                analysis['task_performance']['classification'] = {
                    'f1_score': score,
                    'accuracy': results.get('accuracy', 0.0),
                    'status': 'excellent' if score > 0.8 else 'good' if score > 0.6 else 'needs_improvement'
                }
                scores.append(score)
            
            elif task == 'content_ranking':
                score = results.get('ndcg_at_5', 0.0)
                analysis['task_performance']['ranking'] = {
                    'ndcg_at_5': score,
                    'content_quality': results.get('overall_content_quality', 0.0),
                    'status': 'excellent' if score > 0.8 else 'good' if score > 0.6 else 'needs_improvement'
                }
                scores.append(score)
            
            elif task == 'technical_seo':
                score = results.get('overall_technical_score', 0.0)
                analysis['task_performance']['technical'] = {
                    'overall_score': score,
                    'status': 'excellent' if score > 0.8 else 'good' if score > 0.6 else 'needs_improvement'
                }
                scores.append(score)
        
        # Calculate overall score
        analysis['overall_score'] = np.mean(scores) if scores else 0.0
        
        # Identify strengths and weaknesses
        for task, perf in analysis['task_performance'].items():
            if perf['status'] == 'excellent':
                analysis['strengths'].append(f"{task.capitalize()} performance is excellent")
            elif perf['status'] == 'needs_improvement':
                analysis['weaknesses'].append(f"{task.capitalize()} needs significant improvement")
        
        return analysis
    
    def _generate_seo_insights(self) -> Dict[str, Any]:
        """Generate SEO-specific insights."""
        insights = {
            'content_optimization': {},
            'user_experience': {},
            'technical_performance': {},
            'ranking_potential': {}
        }
        
        if 'content_ranking' in self.results:
            content_results = self.results['content_ranking']
            
            # Content optimization insights
            content_quality = content_results.get('overall_content_quality', 0.0)
            insights['content_optimization'] = {
                'current_score': content_quality,
                'recommendation': 'Focus on content length and readability' if content_quality < 0.7 else 'Content quality is good',
                'priority': 'high' if content_quality < 0.6 else 'medium' if content_quality < 0.8 else 'low'
            }
            
            # User experience insights
            engagement_score = content_results.get('overall_engagement_score', 0.0)
            insights['user_experience'] = {
                'current_score': engagement_score,
                'recommendation': 'Improve page load speed and mobile experience' if engagement_score < 0.6 else 'User experience is good',
                'priority': 'high' if engagement_score < 0.5 else 'medium' if engagement_score < 0.7 else 'low'
            }
        
        if 'technical_seo' in self.results:
            tech_results = self.results['technical_seo']
            
            # Technical performance insights
            tech_score = tech_results.get('overall_technical_score', 0.0)
            insights['technical_performance'] = {
                'current_score': tech_score,
                'recommendation': 'Address Core Web Vitals and mobile optimization' if tech_score < 0.7 else 'Technical performance is good',
                'priority': 'high' if tech_score < 0.6 else 'medium' if tech_score < 0.8 else 'low'
            }
        
        # Ranking potential insights
        if 'content_ranking' in self.results:
            ranking_score = self.results['content_ranking'].get('ndcg_at_5', 0.0)
            insights['ranking_potential'] = {
                'current_score': ranking_score,
                'recommendation': 'Improve content relevance and user signals' if ranking_score < 0.7 else 'Good ranking potential',
                'priority': 'high' if ranking_score < 0.6 else 'medium' if ranking_score < 0.8 else 'low'
            }
        
        return insights
    
    def _suggest_next_steps(self) -> List[str]:
        """Suggest next steps for improvement."""
        next_steps = []
        
        if not self.results:
            next_steps.append("Complete model evaluation to identify improvement areas")
            return next_steps
        
        # Analyze results and suggest steps
        if 'keyword_classification' in self.results:
            f1_score = self.results['keyword_classification'].get('f1_score', 0.0)
            if f1_score < 0.7:
                next_steps.append("Improve keyword classification model with data augmentation")
                next_steps.append("Fine-tune hyperparameters and model architecture")
        
        if 'content_ranking' in self.results:
            content_quality = self.results['content_ranking'].get('overall_content_quality', 0.0)
            if content_quality < 0.7:
                next_steps.append("Optimize content quality metrics (length, readability, keyword density)")
                next_steps.append("Implement content scoring algorithms")
        
        if 'technical_seo' in self.results:
            tech_score = self.results['technical_seo'].get('overall_technical_score', 0.0)
            if tech_score < 0.7:
                next_steps.append("Optimize Core Web Vitals (LCP, FID, CLS)")
                next_steps.append("Improve mobile friendliness and page load speed")
        
        # General next steps
        next_steps.append("Implement A/B testing for model improvements")
        next_steps.append("Set up continuous monitoring and evaluation pipeline")
        next_steps.append("Document model performance and improvement strategies")
        
        return next_steps
    
    def save_evaluation_report(self, filepath: str):
        """Save evaluation report to file."""
        try:
            report = self.generate_evaluation_report()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✅ Evaluation report saved to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")

async def main():
    """Main demonstration function."""
    print("🚀 SEO Model Evaluation Example")
    print("=" * 50)
    
    # Create example evaluator
    evaluator_example = SEOModelEvaluatorExample()
    
    # Setup evaluation system
    await evaluator_example.setup_evaluation()
    
    # Create sample models
    keyword_model = SEOKeywordClassifier(input_dim=768, num_classes=5)
    ranking_model = SEORankingModel(input_dim=768)
    
    # Create test data
    print("\n📊 Creating test data...")
    test_data = create_seo_test_data(n_samples=1000, n_classes=5)
    
    # Evaluate keyword classification
    await evaluator_example.evaluate_keyword_classification(keyword_model, test_data)
    
    # Evaluate content ranking
    await evaluator_example.evaluate_content_ranking(ranking_model, test_data)
    
    # Evaluate technical SEO
    await evaluator_example.evaluate_technical_seo(test_data)
    
    # Generate and display report
    print("\n📋 Generating Evaluation Report...")
    report = evaluator_example.generate_evaluation_report()
    
    # Display summary
    print(f"\n🎯 Overall Model Performance: {report['model_performance']['overall_score']:.4f}")
    
    # Display insights
    print(f"\n💡 SEO Insights:")
    for category, insights in report['seo_insights'].items():
        if insights:
            print(f"  {category.replace('_', ' ').title()}:")
            for key, value in insights.items():
                if key != 'priority':
                    print(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Display recommendations
    if report['improvement_recommendations']:
        print(f"\n🔧 Improvement Recommendations:")
        for i, rec in enumerate(report['improvement_recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Display next steps
    if report['next_steps']:
        print(f"\n📈 Next Steps:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"  {i}. {step}")
    
    # Save report
    evaluator_example.save_evaluation_report("seo_evaluation_report.json")
    
    # Create visualizations if evaluator is available
    if evaluator_example.evaluator:
        print(f"\n📊 Creating evaluation visualizations...")
        evaluator_example.evaluator.plot_evaluation_metrics("seo_evaluation_plots.png")
    
    print(f"\n✅ SEO evaluation example completed successfully!")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

