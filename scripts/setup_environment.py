#!/usr/bin/env python3
"""
Setup script for scam detection bot environment
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command):
    """Run shell command and return result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def setup_environment():
    """Setup the development environment"""
    print("🚀 Setting up Scam Detection Bot environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print("✅ Python version check passed")
    
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        print("📦 Creating virtual environment...")
        success, output = run_command("python -m venv venv")
        if not success:
            print(f"❌ Failed to create virtual environment: {output}")
            sys.exit(1)
        print("✅ Virtual environment created")
    
    # Install requirements
    print("📚 Installing requirements...")
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    success, output = run_command(f"{pip_cmd} install -r requirements.txt")
    if not success:
        print(f"❌ Failed to install requirements: {output}")
        sys.exit(1)
    print("✅ Requirements installed")
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        print("⚙️ Creating .env file...")
        env_content = """GROQ_API_KEY=your_groq_api_key_here
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env file created")
        print("⚠️  Please update .env file with your actual Groq API key")
    
    # Create necessary directories
    directories = [
        "ml_models/trained_models",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Update .env file with your Groq API key")
    print("2. Run: python run.py")
    print("3. Open: http://localhost:8000")

if __name__ == "__main__":
    setup_environment()
