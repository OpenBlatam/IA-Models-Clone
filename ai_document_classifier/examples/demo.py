"""
Demo script for AI Document Classifier
======================================

This script demonstrates the functionality of the AI Document Classifier
by testing various document queries and showing template exports.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import the classifier
sys.path.append(str(Path(__file__).parent.parent))

from document_classifier_engine import DocumentClassifierEngine, DocumentType

def main():
    """Run the demo"""
    print("🤖 AI Document Type Classifier - Demo")
    print("=" * 50)
    
    # Initialize the classifier
    classifier = DocumentClassifierEngine()
    
    # Test queries with different document types
    test_queries = [
        {
            "query": "I want to write a science fiction novel about space exploration and alien civilizations",
            "expected": "novel"
        },
        {
            "query": "Create a service agreement contract for web development services with payment terms",
            "expected": "contract"
        },
        {
            "query": "Design a new mobile app interface with user experience focus and technical specifications",
            "expected": "design"
        },
        {
            "query": "Write a comprehensive business plan for a tech startup including market analysis and financial projections",
            "expected": "business_plan"
        },
        {
            "query": "Research paper on machine learning algorithms with methodology and experimental results",
            "expected": "academic_paper"
        },
        {
            "query": "User manual for a new software application with installation instructions and troubleshooting guide",
            "expected": "user_manual"
        },
        {
            "query": "Marketing campaign materials for a new product launch with promotional content and call-to-action",
            "expected": "marketing_material"
        },
        {
            "query": "Technical manual for industrial equipment with safety procedures and maintenance instructions",
            "expected": "technical_manual"
        },
        {
            "query": "Quarterly business report with performance metrics and strategic recommendations",
            "expected": "report"
        },
        {
            "query": "Project proposal for implementing a new IT system with budget and timeline",
            "expected": "proposal"
        }
    ]
    
    print(f"\n📝 Testing {len(test_queries)} document classification queries...\n")
    
    correct_predictions = 0
    total_predictions = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        
        print(f"Test {i}: {query[:60]}...")
        
        # Classify the document
        result = classifier.classify_document(query, use_ai=False)
        
        # Check if prediction is correct
        is_correct = result.document_type.value == expected
        if is_correct:
            correct_predictions += 1
        
        print(f"  📊 Predicted: {result.document_type.value}")
        print(f"  🎯 Expected:  {expected}")
        print(f"  ✅ Correct:   {'Yes' if is_correct else 'No'}")
        print(f"  🎯 Confidence: {result.confidence:.2f}")
        print(f"  🔑 Keywords: {', '.join(result.keywords[:3])}")
        print(f"  💭 Reasoning: {result.reasoning}")
        
        # Show available templates
        if result.template_suggestions:
            print(f"  📋 Templates: {', '.join(result.template_suggestions)}")
            
            # Export first template as example
            templates = classifier.get_templates(result.document_type)
            if templates:
                template = templates[0]
                print(f"  📄 Sample Template Export ({template.name}):")
                exported = classifier.export_template(template, "markdown")
                # Show first few lines of export
                lines = exported.split('\n')[:5]
                for line in lines:
                    print(f"    {line}")
                print("    ...")
        
        print("-" * 50)
    
    # Summary
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\n📈 Classification Results:")
    print(f"  Total Tests: {total_predictions}")
    print(f"  Correct: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.1f}%")
    
    # Show all available document types
    print(f"\n📚 Supported Document Types:")
    for doc_type in DocumentType:
        if doc_type != DocumentType.UNKNOWN:
            templates = classifier.get_templates(doc_type)
            print(f"  • {doc_type.value}: {len(templates)} templates available")
    
    # Interactive mode
    print(f"\n🎮 Interactive Mode")
    print("Enter document descriptions to classify (type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n📝 Enter document description: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Classify user input
            result = classifier.classify_document(user_input, use_ai=False)
            
            print(f"\n🎯 Classification Result:")
            print(f"  Document Type: {result.document_type.value}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Keywords: {', '.join(result.keywords)}")
            print(f"  Reasoning: {result.reasoning}")
            
            # Show templates
            if result.template_suggestions:
                print(f"\n📋 Available Templates:")
                for template_name in result.template_suggestions:
                    print(f"  • {template_name}")
                
                # Ask if user wants to export a template
                export_choice = input("\n📄 Export a template? (y/n): ").strip().lower()
                if export_choice in ['y', 'yes']:
                    templates = classifier.get_templates(result.document_type)
                    if templates:
                        template = templates[0]
                        format_choice = input("Choose format (json/yaml/markdown): ").strip().lower()
                        if format_choice in ['json', 'yaml', 'markdown']:
                            exported = classifier.export_template(template, format_choice)
                            print(f"\n📄 Exported Template ({template.name}):")
                            print("-" * 30)
                            print(exported)
                            print("-" * 30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Demo completed. Thank you!")

if __name__ == "__main__":
    main()



























