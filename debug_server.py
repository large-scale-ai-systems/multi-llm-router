#!/usr/bin/env python3
"""
Debug Helper Script for LLM Router FastAPI Server

This script provides various debugging utilities and configurations
for developing and troubleshooting the LLM Router FastAPI application.

Usage:
    python debug_server.py [options]

Options:
    --mode debug      : Start server in debug mode with verbose logging
    --mode test       : Start server with test configuration
    --mode profile    : Start server with performance profiling
    --port PORT       : Override default port (8000)
    --reload          : Enable auto-reload on code changes
    --log-level LEVEL : Set log level (debug, info, warning, error)
"""

import os
import sys
import argparse
import logging
import uvicorn
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_debug_logging():
    """Configure detailed debug logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug_server.log')
        ]
    )

def setup_test_environment():
    """Set up environment for testing"""
    os.environ['DEBUG'] = 'true'
    os.environ['TESTING'] = 'true'
    print("🧪 Test environment configured")

def setup_profiling():
    """Set up performance profiling"""
    try:
        import cProfile
        import pstats
        print("Profiling enabled - performance data will be collected")
        os.environ['PROFILING'] = 'true'
    except ImportError:
        print("Warning: cProfile not available - profiling disabled")

def print_debug_info():
    """Print debugging information"""
    print("\n" + "="*60)
    print("🐛 LLM ROUTER DEBUG SERVER")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📦 Python version: {sys.version.split()[0]}")
    print(f"📂 Working directory: {os.getcwd()}")
    print(f"Config file: {project_root / 'config.ini'}")
    print("="*60)
    
    # Check if config file exists
    config_path = project_root / 'config.ini'
    if config_path.exists():
        print("Configuration file found")
    else:
        print("Error: Configuration file missing!")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual environment detected")
    else:
        print("Warning: Not running in virtual environment")
    
    print("="*60)
    print("\nDEBUGGING TIPS:")
    print("1. Set breakpoints in VS Code by clicking next to line numbers")
    print("2. Use F5 to start debugging with 'Debug FastAPI Server' configuration")
    print("3. Use F10 to step over, F11 to step into functions")
    print("4. Check the Debug Console for variable inspection")
    print("5. Use the integrated terminal for interactive debugging")
    print("\n📍 COMMON BREAKPOINT LOCATIONS:")
    print("• fastapi_app.py:155 - Router initialization")
    print("• fastapi_app.py:300 - Query processing") 
    print("• main.py:165 - Provider creation")
    print("• routers/vector_enhanced_router.py - Query routing logic")
    print("\nDEBUG ENDPOINTS:")
    print("• GET  http://localhost:8000/health - Health check")
    print("• POST http://localhost:8000/init - Initialize system") 
    print("• GET  http://localhost:8000/stats - System statistics")
    print("• GET  http://localhost:8000/docs - API documentation")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Debug FastAPI LLM Router Server")
    parser.add_argument('--mode', choices=['debug', 'test', 'profile'], default='debug',
                       help='Debug mode to use')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run server on')
    parser.add_argument('--reload', action='store_true', default=True,
                       help='Enable auto-reload')
    parser.add_argument('--log-level', choices=['debug', 'info', 'warning', 'error'], 
                       default='debug', help='Logging level')
    
    args = parser.parse_args()
    
    # Set debug mode
    os.environ['DEBUG'] = 'true'
    
    # Configure based on mode
    if args.mode == 'debug':
        setup_debug_logging()
        print_debug_info()
    elif args.mode == 'test':
        setup_test_environment()
        setup_debug_logging()
    elif args.mode == 'profile':
        setup_profiling()
        setup_debug_logging()
    
    print(f"Starting FastAPI server in {args.mode.upper()} mode...")
    print(f"🌐 Server will be available at: http://localhost:{args.port}")
    print(f"📚 API Documentation: http://localhost:{args.port}/docs")
    print("\nPress Ctrl+C to stop the server")
    print("🐛 Use VS Code debugger for breakpoints and step-through debugging\n")
    
    try:
        # Import and run the FastAPI app
        from fastapi_app import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True,
            reload_dirs=[str(project_root)],
            reload_excludes=['*.log', '__pycache__', '.git']
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\nError: Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()