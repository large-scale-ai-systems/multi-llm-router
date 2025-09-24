#!/usr/bin/env python3
"""
LLM Router FastAPI Server Startup Script

This script starts the FastAPI server with proper configuration
and provides helpful startup information.
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['fastapi', 'uvicorn', 'pydantic']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Error: Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install fastapi uvicorn pydantic")
        return False
    
    return True


def check_config_file():
    """Check if configuration file exists"""
    config_file = Path("config.ini")
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        print("Please create config.ini based on config.sample.ini")
        return False
    
    print(f"Configuration file found: {config_file}")
    return True


def check_data_files():
    """Check if required data files exist"""
    data_file = Path("data/human_eval_set.csv")
    if not data_file.exists():
        print(f"Warning: Evaluation data file not found: {data_file}")
        print("The system will work but vector evaluation may be limited")
        return False
    
    print(f"Evaluation data file found: {data_file}")
    return True


def main():
    """Main startup routine"""
    print("Starting LLM Router FastAPI Server")
    print("=" * 40)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Working directory: {current_dir}")
    
    # Check dependencies
    print("\n📦 Checking Dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("All dependencies are available")
    
    # Check configuration
    print("\n⚙️  Checking Configuration...")
    if not check_config_file():
        sys.exit(1)
    
    # Check data files
    print("\nChecking Data Files...")
    check_data_files()  # Warning only, not fatal
    
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    print(f"\n🌐 Server Configuration:")
    print(f"  • Host: {host}")
    print(f"  • Port: {port}")
    print(f"  • Reload: {reload}")
    print(f"  • Log Level: {log_level}")
    
    print(f"\n📡 API Endpoints will be available at:")
    print(f"  • Health:     http://{host}:{port}/health")
    print(f"  • Initialize: http://{host}:{port}/init")
    print(f"  • Query:      http://{host}:{port}/query")
    print(f"  • Stats:      http://{host}:{port}/stats")
    print(f"  • Docs:       http://{host}:{port}/docs")
    
    print(f"\nStarting server...")
    print("=" * 40)
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "fastapi_app:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n\nError: Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()