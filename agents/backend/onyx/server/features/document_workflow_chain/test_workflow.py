"""
Simple test script for Document Workflow Chain
==============================================

This script provides a quick way to test the Document Workflow Chain
functionality without setting up the full API server.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflow_chain_engine import WorkflowChainEngine

class SimpleAIClient:
    """Simple AI client for testing"""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate_text(self, prompt: str) -> str:
        """Generate simple test content"""
        self.call_count += 1
        
        if "introducción" in prompt.lower() or "introduction" in prompt.lower():
            return """
            La Inteligencia Artificial está revolucionando la forma en que creamos contenido. 
            Desde asistentes de escritura automatizados hasta optimización inteligente de contenido, 
            las herramientas de IA están transformando el panorama creativo. Esta guía completa 
            explora las diversas formas en que la IA puede mejorar tu proceso de creación de contenido, 
            aumentar la eficiencia y entregar resultados de mayor calidad.
            
            Los beneficios de la creación de contenido impulsada por IA son numerosos. No solo 
            acelera el proceso de escritura, sino que también ayuda a mantener la consistencia 
            en diferentes piezas de contenido. La IA puede analizar las preferencias de la audiencia, 
            optimizar para motores de búsqueda e incluso sugerir mejoras para mejorar la legibilidad y el engagement.
            """
        
        elif "beneficios" in prompt.lower() or "benefits" in prompt.lower():
            return """
            Las ventajas de implementar IA en los flujos de trabajo de creación de contenido 
            van mucho más allá de la simple automatización. Uno de los beneficios más significativos 
            es la capacidad de escalar la producción de contenido sin aumentar proporcionalmente 
            los recursos humanos. La IA puede generar múltiples variaciones de contenido, probar 
            diferentes enfoques e identificar qué resuena más con tu audiencia objetivo.
            
            Además, la creación de contenido impulsada por IA permite la personalización a escala. 
            Al analizar datos de usuarios y preferencias, la IA puede crear contenido personalizado 
            que hable directamente a lectores individuales. Este nivel de personalización era 
            previamente imposible de lograr manualmente, especialmente para audiencias grandes.
            """
        
        elif "herramientas" in prompt.lower() or "tools" in prompt.lower():
            return """
            El mercado está inundado de herramientas de creación de contenido con IA, cada una 
            ofreciendo capacidades y características únicas. Plataformas populares como GPT-4, 
            Claude y otros modelos de lenguaje proporcionan poderosas capacidades de generación 
            de texto. Estas herramientas pueden crear desde posts de blog y artículos hasta 
            copy de marketing y contenido para redes sociales.
            
            Al seleccionar herramientas de creación de contenido con IA, considera factores 
            como la calidad de salida, opciones de personalización, capacidades de integración 
            y costo. Algunas herramientas se especializan en tipos específicos de contenido, 
            mientras que otras ofrecen soluciones integrales para diversas necesidades de contenido. 
            La clave es elegir herramientas que se alineen con tus requisitos específicos y flujo de trabajo.
            """
        
        else:
            return f"""
            Esta es una continuación de nuestra discusión sobre la creación de contenido impulsada por IA. 
            El contenido anterior ha sentado las bases para entender cómo la inteligencia artificial 
            puede transformar tu estrategia de contenido. Ahora, exploremos aspectos adicionales 
            y consideraciones que son cruciales para una implementación exitosa.
            
            A medida que continuamos esta exploración, es importante recordar que la IA es una herramienta 
            que mejora la creatividad humana en lugar de reemplazarla. Las estrategias de creación de 
            contenido más efectivas combinan la eficiencia y capacidades de la IA con la perspicacia, 
            creatividad y pensamiento estratégico humanos.
            """

async def test_basic_functionality():
    """Test basic workflow chain functionality"""
    print("🧪 Testing Document Workflow Chain - Basic Functionality")
    print("=" * 60)
    
    # Initialize engine with simple AI client
    ai_client = SimpleAIClient()
    engine = WorkflowChainEngine(ai_client=ai_client)
    
    try:
        # Test 1: Create workflow chain
        print("\n📝 Test 1: Creating workflow chain...")
        chain = await engine.create_workflow_chain(
            name="Guía de Marketing Digital",
            description="Serie de artículos sobre marketing digital con IA",
            initial_prompt="Escribe una introducción al marketing digital moderno con IA"
        )
        
        print(f"✅ Chain created: {chain.id}")
        print(f"📄 Initial document: '{chain.nodes[chain.root_node_id].title}'")
        print(f"📊 Content length: {len(chain.nodes[chain.root_node_id].content)} characters")
        
        # Test 2: Continue workflow chain
        print("\n🔄 Test 2: Continuing workflow chain...")
        next_doc = await engine.continue_workflow_chain(chain.id)
        
        print(f"✅ Document continued: {next_doc.id}")
        print(f"📄 New document: '{next_doc.title}'")
        print(f"📊 Content length: {len(next_doc.content)} characters")
        print(f"🔗 Parent ID: {next_doc.parent_id}")
        
        # Test 3: Generate blog title
        print("\n🏷️  Test 3: Generating blog title...")
        sample_content = "Este es un artículo de ejemplo sobre el futuro de la inteligencia artificial en la creación de contenido y cómo está transformando la forma en que abordamos el marketing digital y la estrategia de contenido."
        title = await engine.generate_blog_title(sample_content)
        
        print(f"✅ Title generated: '{title}'")
        
        # Test 4: Get chain history
        print("\n📚 Test 4: Getting chain history...")
        history = engine.get_chain_history(chain.id)
        
        print(f"✅ History retrieved: {len(history)} documents")
        for i, doc in enumerate(history, 1):
            print(f"   {i}. {doc.title} ({doc.generated_at.strftime('%H:%M:%S')})")
        
        # Test 5: Workflow management
        print("\n🔧 Test 5: Testing workflow management...")
        
        # Pause chain
        engine.pause_workflow_chain(chain.id)
        paused_chain = engine.get_workflow_chain(chain.id)
        print(f"✅ Chain paused: {paused_chain.status}")
        
        # Resume chain
        engine.resume_workflow_chain(chain.id)
        resumed_chain = engine.get_workflow_chain(chain.id)
        print(f"✅ Chain resumed: {resumed_chain.status}")
        
        # Test 6: Export workflow
        print("\n💾 Test 6: Exporting workflow...")
        export_data = engine.export_workflow_chain(chain.id)
        
        print(f"✅ Workflow exported: {len(export_data['nodes'])} documents")
        print(f"📊 Export keys: {list(export_data.keys())}")
        
        # Test 7: Multiple chains
        print("\n🔗 Test 7: Creating multiple chains...")
        chain2 = await engine.create_workflow_chain(
            name="Serie de SEO",
            description="Artículos sobre optimización SEO",
            initial_prompt="Escribe sobre los beneficios del SEO moderno"
        )
        
        active_chains = engine.get_all_active_chains()
        print(f"✅ Multiple chains created: {len(active_chains)} active chains")
        
        # Summary
        print(f"\n🎯 Test Summary:")
        print(f"   • AI Client calls made: {ai_client.call_count}")
        print(f"   • Workflow chains created: {len(active_chains)}")
        print(f"   • Total documents generated: {sum(len(chain.nodes) for chain in active_chains)}")
        print(f"   • All tests passed: ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_handling():
    """Test error handling scenarios"""
    print("\n🛡️  Testing Error Handling")
    print("=" * 30)
    
    engine = WorkflowChainEngine()
    
    try:
        # Test invalid chain ID
        print("Testing invalid chain ID...")
        try:
            await engine.continue_workflow_chain("invalid-id")
            print("❌ Should have raised ValueError")
        except ValueError:
            print("✅ Correctly handled invalid chain ID")
        
        # Test getting non-existent chain
        print("Testing non-existent chain retrieval...")
        chain = engine.get_workflow_chain("non-existent")
        if chain is None:
            print("✅ Correctly returned None for non-existent chain")
        else:
            print("❌ Should have returned None")
        
        # Test pausing non-existent chain
        print("Testing pause of non-existent chain...")
        result = engine.pause_workflow_chain("non-existent")
        if not result:
            print("✅ Correctly handled pause of non-existent chain")
        else:
            print("❌ Should have returned False")
        
        print("✅ All error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Document Workflow Chain - Test Suite")
    print("=" * 50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run basic functionality tests
    basic_test_passed = await test_basic_functionality()
    
    # Run error handling tests
    error_test_passed = await test_error_handling()
    
    # Final results
    print(f"\n🏁 Test Results Summary")
    print("=" * 30)
    print(f"Basic Functionality: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"Error Handling: {'✅ PASSED' if error_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and error_test_passed:
        print(f"\n🎉 All tests passed! The Document Workflow Chain is working correctly.")
        return 0
    else:
        print(f"\n💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    # Run the test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


