#!/usr/bin/env python3
"""
TruthGPT API Server Startup Script
===================================

Simple script to start the TruthGPT API server.
"""

import argparse

def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(
        description="Start TruthGPT API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_server.py
  python start_server.py --host 0.0.0.0 --port 8000
  python start_server.py --host localhost --port 3000 --reload
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TruthGPT API Server")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    print("=" * 60)
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/health")
    print("=" * 60)
    print()
    
    try:
        import uvicorn
        from api.main import app
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1
        )
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()











