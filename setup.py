#!/usr/bin/env python3
"""
FlexoTwin Smart Maintenance 4.0 - Setup Script
Automated setup untuk development environment
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

class FlexoTwinSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.venv_path = self.project_root / "venv"
        
    def create_virtual_environment(self):
        """Create virtual environment"""
        print("🔧 Creating virtual environment...")
        
        if self.venv_path.exists():
            print("   ✅ Virtual environment already exists")
            return True
            
        try:
            venv.create(self.venv_path, with_pip=True)
            print("   ✅ Virtual environment created successfully")
            return True
        except Exception as e:
            print(f"   ❌ Error creating virtual environment: {str(e)}")
            return False
    
    def install_requirements(self):
        """Install Python packages"""
        print("📦 Installing requirements...")
        
        # Determine pip executable path
        if sys.platform == "win32":
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
            python_exe = self.venv_path / "Scripts" / "python.exe"
        else:
            pip_exe = self.venv_path / "bin" / "pip"
            python_exe = self.venv_path / "bin" / "python"
        
        if not pip_exe.exists():
            print(f"   ❌ Pip not found at {pip_exe}")
            return False
            
        try:
            # Upgrade pip first
            subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], 
                          check=True, capture_output=True)
            print("   ✅ Pip upgraded")
            
            # Install requirements
            subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], 
                          check=True, capture_output=True)
            print("   ✅ Requirements installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error installing requirements: {str(e)}")
            return False
    
    def check_project_structure(self):
        """Verify project structure"""
        print("📂 Checking project structure...")
        
        required_dirs = [
            "01_Scripts",
            "02_Models", 
            "03_Data",
            "04_Visualizations",
            "05_API",
            "06_Documentation",
            "07_Examples"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                dir_path.mkdir(exist_ok=True)
                print(f"   📁 Created directory: {dir_name}")
            else:
                print(f"   ✅ Found directory: {dir_name}")
        
        return len(missing_dirs) == 0
    
    def create_activation_script(self):
        """Create activation script"""
        print("🚀 Creating activation script...")
        
        if sys.platform == "win32":
            script_content = """@echo off
echo 🎯 FlexoTwin Smart Maintenance 4.0
echo =====================================
echo Activating virtual environment...

call venv\\Scripts\\activate.bat

echo ✅ Environment activated!
echo.
echo Quick Commands:
echo   python 05_API\\07_api_interface.py  - Start API server
echo   python 05_API\\simple_api_test.py   - Test API
echo   python 01_Scripts\\01_data_exploration.py - Run analysis
echo.
echo 📚 Documentation: 06_Documentation\\README.md
echo 🌐 API Docs: http://localhost:5000/api/docs (when server running)
echo.

cmd /k
"""
            script_path = self.project_root / "activate.bat"
        else:
            script_content = """#!/bin/bash
echo "🎯 FlexoTwin Smart Maintenance 4.0"
echo "====================================="
echo "Activating virtual environment..."

source venv/bin/activate

echo "✅ Environment activated!"
echo ""
echo "Quick Commands:"
echo "  python 05_API/07_api_interface.py  - Start API server"
echo "  python 05_API/simple_api_test.py   - Test API"
echo "  python 01_Scripts/01_data_exploration.py - Run analysis"
echo ""
echo "📚 Documentation: 06_Documentation/README.md"
echo "🌐 API Docs: http://localhost:5000/api/docs (when server running)"
echo ""

bash
"""
            script_path = self.project_root / "activate.sh"
            
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
                
            if sys.platform != "win32":
                os.chmod(script_path, 0o755)
                
            print(f"   ✅ Activation script created: {script_path.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating activation script: {str(e)}")
            return False
    
    def run_setup(self):
        """Run complete setup process"""
        print("🎯 FlexoTwin Smart Maintenance 4.0 - Setup")
        print("=" * 50)
        
        steps = [
            ("Create Virtual Environment", self.create_virtual_environment),
            ("Install Requirements", self.install_requirements),
            ("Check Project Structure", self.check_project_structure),
            ("Create Activation Script", self.create_activation_script)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            success = step_func()
            if not success:
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        print("\n" + "=" * 50)
        print("🎉 Setup completed successfully!")
        print("")
        print("Next steps:")
        if sys.platform == "win32":
            print("  1. Run: activate.bat")
        else:
            print("  1. Run: ./activate.sh")
        print("  2. Start API: python 05_API/07_api_interface.py")
        print("  3. Test API: python 05_API/simple_api_test.py")
        print("")
        print("📚 Check README.md for detailed instructions")
        
        return True

if __name__ == "__main__":
    setup = FlexoTwinSetup()
    success = setup.run_setup()
    sys.exit(0 if success else 1)