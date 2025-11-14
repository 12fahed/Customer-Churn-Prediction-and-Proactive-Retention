"""
Quick Start Script for GRAHAK CRM System
Run this script to set up everything automatically
"""

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")

def run_command(command, description):
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully!")
            return True
        else:
            print(f"✗ {description} failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def main():
    print_header("GRAHAK CRM - Quick Start Setup")
    
    print("This script will:")
    print("1. Check if Python is installed")
    print("2. Install required packages")
    print("3. Create models directory")
    print("4. Train optimized models")
    print("5. Launch the application")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check Python
    print_header("Step 1: Checking Python Installation")
    python_version = sys.version_info
    print(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("✗ Python 3.7 or higher is required!")
        sys.exit(1)
    print("✓ Python version is compatible")
    
    # Step 2: Install packages
    print_header("Step 2: Installing Required Packages")
    if run_command("pip install -r requirements.txt", "Installing packages"):
        print("\nInstalled packages:")
        print("- streamlit (Web framework)")
        print("- pandas, numpy (Data processing)")
        print("- scikit-learn (Machine learning)")
        print("- plotly (Visualizations)")
        print("- joblib (Model persistence)")
        print("- mlxtend, deap, textblob (Advanced features)")
    else:
        print("\n⚠ Package installation failed. Please run manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 3: Create models directory
    print_header("Step 3: Creating Models Directory")
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ Models directory created")
    else:
        print("✓ Models directory already exists")
    
    # Step 4: Check for data files
    print_header("Step 4: Checking Data Files")
    data_files = ['data/train.csv', 'data/test.csv']
    all_data_exists = True
    
    for file in data_files:
        if os.path.exists(file):
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")
            all_data_exists = False
    
    if not all_data_exists:
        print("\n⚠ Warning: Some data files are missing!")
        print("Please ensure CSV files are in the data/ directory")
        proceed = input("\nDo you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    # Step 5: Train models
    print_header("Step 5: Training Optimized Models")
    
    # Check if models already exist
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')] if os.path.exists('models') else []
    
    if model_files:
        print(f"✓ Found {len(model_files)} existing model file(s)")
        retrain = input("\nDo you want to retrain models? (y/n): ")
        if retrain.lower() == 'y':
            run_command("python train_models.py", "Training models")
    else:
        print("No existing models found. Training new models...")
        if not run_command("python train_models.py", "Training models"):
            print("\n⚠ Model training failed. You can:")
            print("1. Train models later using: python train_models.py")
            print("2. Use the system to train models interactively")
    
    # Step 6: Launch application
    print_header("Step 6: Launching GRAHAK CRM System")
    print("\nThe application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nTo stop the application, press Ctrl+C in this terminal")
    
    input("\nPress Enter to launch...")
    
    print("\nStarting Streamlit application...")
    print("=" * 70)
    
    # Launch streamlit
    subprocess.run("streamlit run script.py", shell=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")
        sys.exit(1)
