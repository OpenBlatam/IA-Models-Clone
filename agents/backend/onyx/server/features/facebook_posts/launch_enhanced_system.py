#!/usr/bin/env python3
"""
Enhanced Facebook Content Optimization System - Launcher
This script launches the enhanced system with the Gradio interface
"""

import sys
import os

def main():
    """Launch the enhanced system"""
    print("🚀 Enhanced Facebook Content Optimization System v2.0.0")
    print("=" * 60)
    
    try:
        # Import and launch the enhanced system
        from enhanced_gradio_interface import EnhancedGradioInterface
        
        print("✅ Enhanced system imported successfully!")
        print("🖥️ Launching Gradio interface...")
        print("\n📋 System Features:")
        print("   - Content Optimization with AI agents")
        print("   - Performance monitoring and analytics")
        print("   - Real-time system health checks")
        print("   - Advanced caching and optimization")
        print("   - Multi-agent consensus decision making")
        
        print("\n🌐 The interface will open in your browser...")
        print("   - Content Optimization Tab: Optimize your Facebook content")
        print("   - AI Agents Tab: Monitor AI agent performance and decisions")
        print("   - Performance Monitoring Tab: Track system performance metrics")
        print("   - System Health Tab: Monitor system health and status")
        print("   - Analytics Tab: View detailed analytics and trends")
        
        # Create and launch the interface
        interface = EnhancedGradioInterface()
        interface.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7861,        # Use different port to avoid conflicts
            share=False,             # Set to True to create a public link
            debug=True,              # Enable debug mode
            show_error=True,         # Show error details
            quiet=False              # Show all output
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("   pip install -r requirements_enhanced_system.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error launching system: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
