from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
from transformers import (
from typing import List, Dict, Any
import json
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Transformers Examples for Email Sequence AI System

This script demonstrates various ways to use Hugging Face Transformers
for email sequence generation, analysis, and optimization.
"""

    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    pipeline, TextGenerationPipeline, SummarizationPipeline,
    TranslationPipeline, TextClassificationPipeline, ZeroShotClassificationPipeline
)


class EmailSequenceTransformers:
    """Transformers-based email sequence generation and analysis."""
    
    def __init__(self) -> Any:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize pipelines
        self.text_generator = None
        self.summarizer = None
        self.classifier = None
        self.translator = None
        self.zero_shot_classifier = None
        
    def load_text_generation_model(self, model_name: str = "gpt2"):
        """Load a text generation model for email content creation."""
        print(f"Loading text generation model: {model_name}")
        
        try:
            self.text_generator = pipeline(
                "text-generation",
                model=model_name,
                device=self.device,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            print(f"✅ Text generation model loaded: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error loading text generation model: {e}")
            return False
    
    def load_summarization_model(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """Load a summarization model for email content summarization."""
        print(f"Loading summarization model: {model_name}")
        
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=self.device,
                max_length=130,
                min_length=30
            )
            print(f"✅ Summarization model loaded: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error loading summarization model: {e}")
            return False
    
    def load_classification_model(self, model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"):
        """Load a classification model for sentiment analysis."""
        print(f"Loading classification model: {model_name}")
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                device=self.device
            )
            print(f"✅ Classification model loaded: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error loading classification model: {e}")
            return False
    
    def load_translation_model(self, model_name: str = "Helsinki-NLP/opus-mt-en-es"):
        """Load a translation model for multilingual email support."""
        print(f"Loading translation model: {model_name}")
        
        try:
            self.translator = pipeline(
                "translation",
                model=model_name,
                device=self.device
            )
            print(f"✅ Translation model loaded: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error loading translation model: {e}")
            return False
    
    def load_zero_shot_classifier(self, model_name: str = "facebook/bart-large-mnli"):
        """Load a zero-shot classification model for flexible categorization."""
        print(f"Loading zero-shot classification model: {model_name}")
        
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=self.device
            )
            print(f"✅ Zero-shot classification model loaded: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Error loading zero-shot classification model: {e}")
            return False
    
    def generate_email_content(self, prompt: str, max_length: int = 200) -> str:
        """Generate email content using the text generation model."""
        if not self.text_generator:
            print("❌ Text generation model not loaded. Call load_text_generation_model() first.")
            return ""
        
        try:
            result = self.text_generator(prompt, max_length=max_length, num_return_sequences=1)
            return result[0]['generated_text']
        except Exception as e:
            print(f"❌ Error generating email content: {e}")
            return ""
    
    def generate_email_sequence(self, topic: str, target_audience: str, sequence_length: int = 5) -> List[Dict[str, str]]:
        """Generate a complete email sequence."""
        if not self.text_generator:
            print("❌ Text generation model not loaded. Call load_text_generation_model() first.")
            return []
        
        sequence = []
        
        # Email 1: Introduction
        intro_prompt = f"Write a professional email introducing {topic} to {target_audience}. Keep it friendly and engaging."
        intro_content = self.generate_email_content(intro_prompt)
        sequence.append({
            "email_number": 1,
            "subject": f"Introduction: {topic}",
            "content": intro_content,
            "type": "introduction"
        })
        
        # Email 2: Value proposition
        value_prompt = f"Write a follow-up email explaining the value and benefits of {topic} for {target_audience}."
        value_content = self.generate_email_content(value_prompt)
        sequence.append({
            "email_number": 2,
            "subject": f"The Value of {topic}",
            "content": value_content,
            "type": "value_proposition"
        })
        
        # Email 3: Social proof
        proof_prompt = f"Write an email with testimonials and social proof about {topic} for {target_audience}."
        proof_content = self.generate_email_content(proof_prompt)
        sequence.append({
            "email_number": 3,
            "subject": f"What Others Say About {topic}",
            "content": proof_content,
            "type": "social_proof"
        })
        
        # Email 4: Call to action
        cta_prompt = f"Write a compelling call-to-action email for {topic} targeting {target_audience}."
        cta_content = self.generate_email_content(cta_prompt)
        sequence.append({
            "email_number": 4,
            "subject": f"Ready to Get Started with {topic}?",
            "content": cta_content,
            "type": "call_to_action"
        })
        
        # Email 5: Follow-up
        follow_prompt = f"Write a gentle follow-up email for {topic} if the {target_audience} hasn't responded yet."
        follow_content = self.generate_email_content(follow_prompt)
        sequence.append({
            "email_number": 5,
            "subject": f"Quick Follow-up on {topic}",
            "content": follow_content,
            "type": "follow_up"
        })
        
        return sequence
    
    def analyze_email_sentiment(self, email_content: str) -> Dict[str, Any]:
        """Analyze the sentiment of email content."""
        if not self.classifier:
            print("❌ Classification model not loaded. Call load_classification_model() first.")
            return {}
        
        try:
            result = self.classifier(email_content)
            return {
                "content": email_content,
                "sentiment": result[0]['label'],
                "confidence": result[0]['score']
            }
        except Exception as e:
            print(f"❌ Error analyzing sentiment: {e}")
            return {}
    
    def categorize_email(self, email_content: str, categories: List[str]) -> Dict[str, Any]:
        """Categorize email content using zero-shot classification."""
        if not self.zero_shot_classifier:
            print("❌ Zero-shot classification model not loaded. Call load_zero_shot_classifier() first.")
            return {}
        
        try:
            result = self.zero_shot_classifier(email_content, candidate_labels=categories)
            return {
                "content": email_content,
                "categories": result['labels'],
                "scores": result['scores'],
                "top_category": result['labels'][0],
                "top_score": result['scores'][0]
            }
        except Exception as e:
            print(f"❌ Error categorizing email: {e}")
            return {}
    
    def summarize_email(self, email_content: str) -> str:
        """Summarize email content."""
        if not self.summarizer:
            print("❌ Summarization model not loaded. Call load_summarization_model() first.")
            return ""
        
        try:
            result = self.summarizer(email_content, max_length=100, min_length=30)
            return result[0]['summary_text']
        except Exception as e:
            print(f"❌ Error summarizing email: {e}")
            return ""
    
    def translate_email(self, email_content: str, target_language: str = "Spanish") -> str:
        """Translate email content to another language."""
        if not self.translator:
            print("❌ Translation model not loaded. Call load_translation_model() first.")
            return ""
        
        try:
            result = self.translator(email_content)
            return result[0]['translation_text']
        except Exception as e:
            print(f"❌ Error translating email: {e}")
            return ""
    
    def optimize_email_subject_line(self, email_content: str) -> List[str]:
        """Generate optimized subject lines for email content."""
        if not self.text_generator:
            print("❌ Text generation model not loaded. Call load_text_generation_model() first.")
            return []
        
        subject_prompts = [
            f"Generate a compelling subject line for this email: {email_content[:100]}...",
            f"Create an attention-grabbing subject line for: {email_content[:100]}...",
            f"Write a professional subject line for: {email_content[:100]}..."
        ]
        
        subject_lines = []
        for prompt in subject_prompts:
            try:
                result = self.text_generator(prompt, max_length=50, num_return_sequences=1)
                subject_line = result[0]['generated_text'].replace(prompt, "").strip()
                if subject_line:
                    subject_lines.append(subject_line)
            except Exception as e:
                print(f"❌ Error generating subject line: {e}")
        
        return subject_lines


def main():
    """Main demonstration function."""
    print("🚀 Transformers Examples for Email Sequence AI System")
    print("=" * 60)
    
    # Initialize the email sequence transformers
    email_transformer = EmailSequenceTransformers()
    
    # Load models
    print("\n📦 Loading Models...")
    email_transformer.load_text_generation_model()
    email_transformer.load_summarization_model()
    email_transformer.load_classification_model()
    email_transformer.load_zero_shot_classifier()
    
    # Example 1: Generate email sequence
    print("\n📧 Example 1: Generating Email Sequence")
    print("-" * 40)
    
    topic = "AI-Powered Email Marketing Platform"
    target_audience = "small business owners"
    
    sequence = email_transformer.generate_email_sequence(topic, target_audience, 3)
    
    for email in sequence:
        print(f"\nEmail {email['email_number']}: {email['subject']}")
        print(f"Type: {email['type']}")
        print(f"Content: {email['content'][:200]}...")
    
    # Example 2: Analyze sentiment
    print("\n😊 Example 2: Sentiment Analysis")
    print("-" * 40)
    
    sample_email = "We're excited to introduce our new AI-powered email marketing platform that will revolutionize how you connect with your customers!"
    sentiment_result = email_transformer.analyze_email_sentiment(sample_email)
    
    if sentiment_result:
        print(f"Email: {sentiment_result['content']}")
        print(f"Sentiment: {sentiment_result['sentiment']}")
        print(f"Confidence: {sentiment_result['confidence']:.2f}")
    
    # Example 3: Categorize email
    print("\n🏷️ Example 3: Email Categorization")
    print("-" * 40)
    
    categories = ["sales", "marketing", "support", "newsletter", "promotional"]
    categorization_result = email_transformer.categorize_email(sample_email, categories)
    
    if categorization_result:
        print(f"Email: {categorization_result['content']}")
        print(f"Top Category: {categorization_result['top_category']}")
        print(f"Confidence: {categorization_result['top_score']:.2f}")
        print("All Categories:")
        for cat, score in zip(categorization_result['categories'], categorization_result['scores']):
            print(f"  - {cat}: {score:.2f}")
    
    # Example 4: Summarize email
    print("\n📝 Example 4: Email Summarization")
    print("-" * 40)
    
    long_email = """
    Dear valued customer,
    
    We hope this email finds you well. We wanted to take a moment to introduce you to our latest innovation in email marketing technology. Our new AI-powered platform combines cutting-edge machine learning algorithms with intuitive user interface design to deliver unprecedented results for businesses of all sizes.
    
    The platform features advanced segmentation capabilities, automated A/B testing, personalized content generation, and comprehensive analytics that provide deep insights into your email campaign performance. Our customers have reported an average 40% increase in open rates and a 60% improvement in click-through rates within the first month of implementation.
    
    We're offering a special 30-day free trial for new customers, along with personalized onboarding support and dedicated account management. Our team of email marketing experts will work closely with you to ensure you get the most out of our platform.
    
    We'd love to schedule a personalized demo to show you how our platform can transform your email marketing efforts. Please let us know if you're interested in learning more about this exciting opportunity.
    
    Best regards,
    The Email Marketing Team
    """
    
    summary = email_transformer.summarize_email(long_email)
    print(f"Original Email Length: {len(long_email)} characters")
    print(f"Summary: {summary}")
    print(f"Summary Length: {len(summary)} characters")
    
    # Example 5: Generate subject lines
    print("\n📋 Example 5: Subject Line Optimization")
    print("-" * 40)
    
    subject_lines = email_transformer.optimize_email_subject_line(sample_email)
    print("Generated Subject Lines:")
    for i, subject in enumerate(subject_lines, 1):
        print(f"  {i}. {subject}")
    
    print("\n🎉 Transformers examples completed successfully!")
    print("\nNext steps:")
    print("1. Experiment with different models")
    print("2. Fine-tune models on your email data")
    print("3. Integrate with the full Email Sequence AI System")
    print("4. Use the Gradio interface for interactive demos")


match __name__:
    case "__main__":
    main() 