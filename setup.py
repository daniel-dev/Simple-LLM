#!/usr/bin/env python3
"""
🚀 LLM Factory Setup Script

This script helps you set up the LLM Factory project environment quickly and easily.
It handles virtual environment creation, dependency installation, and initial setup.
"""

import sys
import subprocess
import platform
import venv
from pathlib import Path

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(text):
    print(f"{Colors.OKBLUE}🔧 {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and handle errors."""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        print_error(f"Error: {e}")
        return False

def check_python_version():
    """Check if Python version is adequate."""
    print_step("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python {version.major}.{version.minor} detected. Python 3.9+ required.")
        sys.exit(1)
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")

def check_git():
    """Check if git is available."""
    print_step("Checking Git installation...")
    
    try:
        result = run_command("git --version", capture_output=True)
        print_success(f"Git detected: {result}")
        return True
    except Exception:
        print_warning("Git not found. Some features may not work.")
        return False

def create_virtual_environment():
    """Create and activate virtual environment."""
    print_step("Creating virtual environment...")
    
    venv_path = Path("llm-factory-env")
    
    if venv_path.exists():
        print_warning("Virtual environment already exists")
        return venv_path
    
    try:
        venv.create(venv_path, with_pip=True)
        print_success("Virtual environment created successfully")
        return venv_path
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        sys.exit(1)

def get_activation_command(venv_path):
    """Get the command to activate virtual environment."""
    system = platform.system().lower()
    
    if system == "windows":
        return f"{venv_path}\\Scripts\\activate"
    else:
        return f"source {venv_path}/bin/activate"

def install_dependencies(venv_path):
    """Install required dependencies."""
    print_step("Installing dependencies...")
    
    system = platform.system().lower()
    
    if system == "windows":
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Upgrade pip first
    print_step("Upgrading pip...")
    if not run_command(f"{python_path} -m pip install --upgrade pip"):
        print_error("Failed to upgrade pip")
        return False
    
    # Install requirements
    if Path("requirements.txt").exists():
        print_step("Installing project dependencies...")
        if not run_command(f"{pip_path} install -r requirements.txt"):
            print_error("Failed to install dependencies")
            return False
        print_success("Dependencies installed successfully")
    else:
        print_warning("requirements.txt not found, installing basic dependencies...")
        basic_deps = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "datasets>=2.10.0",
            "flask>=2.0.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        ]
        
        for dep in basic_deps:
            print_step(f"Installing {dep}...")
            if not run_command(f"{pip_path} install {dep}"):
                print_error(f"Failed to install {dep}")
                return False
        
        print_success("Basic dependencies installed")
    
    return True

def setup_directories():
    """Create necessary directories."""
    print_step("Setting up project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/cache",
        "models/trained",
        "models/checkpoints",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_success("Project directories created")

def download_sample_data():
    """Download or create sample data if needed."""
    print_step("Checking sample data...")
    
    sample_data_path = Path("data/raw/imdb_sample.csv")
    
    if sample_data_path.exists():
        print_success("Sample data already exists")
        return True
    
    print_step("Sample data not found. You can add your own data to data/raw/")
    return True

def create_config_file():
    """Create a basic configuration file."""
    print_step("Creating configuration file...")
    
    config_content = """# LLM Factory Configuration
# This file contains basic configuration for your LLM Factory setup

# Model Configuration
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Data Configuration
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# Training Configuration
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 100

# Web Interface
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
LOGS_DIR = "logs"
"""
    
    config_path = Path("config.py")
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print_success("Configuration file created")
    else:
        print_warning("Configuration file already exists")

def print_next_steps(venv_path):
    """Print next steps for the user."""
    print_header("🎉 Setup Complete!")    
    activation_cmd = get_activation_command(venv_path)
    
    print(f"{Colors.OKGREEN}Your LLM Factory environment is ready!{Colors.ENDC}\n")
    
    print(f"{Colors.OKBLUE}📋 Next Steps:{Colors.ENDC}")
    print("1. Activate your virtual environment:")
    print(f"   {Colors.OKCYAN}{activation_cmd}{Colors.ENDC}")
    
    print("\n2. Start the web application:")
    print(f"   {Colors.OKCYAN}python app.py{Colors.ENDC}")
    
    print("\n3. Or run the complete pipeline:")
    print(f"   {Colors.OKCYAN}python run_pipeline.py{Colors.ENDC}")
    
    print("\n4. Open your browser and go to:")
    print(f"   {Colors.OKCYAN}http://localhost:5000{Colors.ENDC}")
    
    print(f"\n{Colors.OKBLUE}📚 Useful Commands:{Colors.ENDC}")
    print(f"   {Colors.OKCYAN}python demo.py{Colors.ENDC} - Run a quick demo")
    print(f"   {Colors.OKCYAN}python simple_train.py{Colors.ENDC} - Simple training example")
    print(f"   {Colors.OKCYAN}python show_results.py{Colors.ENDC} - View training results")
    
    print(f"\n{Colors.OKBLUE}📖 Documentation:{Colors.ENDC}")
    print(f"   {Colors.OKCYAN}README.md{Colors.ENDC} - Project overview")
    print(f"   {Colors.OKCYAN}docs/{Colors.ENDC} - Detailed documentation")
    
    print(f"\n{Colors.WARNING}💡 Tips:{Colors.ENDC}")
    print(f"   • Add your data to {Colors.OKCYAN}data/raw/{Colors.ENDC}")
    print(f"   • Modify {Colors.OKCYAN}config.py{Colors.ENDC} for custom settings")
    print(f"   • Check {Colors.OKCYAN}notebooks/{Colors.ENDC} for Jupyter examples")

def main():
    """Main setup function."""
    print_header("🏭 LLM Factory Setup")
    print(f"{Colors.OKBLUE}Welcome to LLM Factory! This script will set up your environment.{Colors.ENDC}\n")
    
    # Check system requirements
    check_python_version()
    check_git()
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    
    # Install dependencies
    if not install_dependencies(venv_path):
        print_error("Setup failed during dependency installation")
        sys.exit(1)
    
    # Setup project structure
    setup_directories()
    download_sample_data()
    create_config_file()
    
    # Show next steps
    print_next_steps(venv_path)
    
    print(f"\n{Colors.OKGREEN}🚀 Happy coding with LLM Factory!{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
