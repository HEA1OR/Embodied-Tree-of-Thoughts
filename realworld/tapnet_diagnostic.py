#!/usr/bin/env python3

"""
TAPnet installation helper and validator
This script helps diagnose and fix TAPnet installation issues
"""

import sys
import os
import subprocess

def check_tapnet_installation():
    """Check if TAPnet is properly installed"""
    print("Checking TAPnet installation...")
    
    try:
        from tapnet.torch1 import tapir_model
        print("✓ TAPnet import successful!")
        return True
    except ImportError as e:
        print(f"✗ TAPnet import failed: {e}")
        return False

def check_checkpoint_exists():
    """Check if the required checkpoint file exists"""
    checkpoint_paths = [
        "checkpoints/causal_bootstapir_checkpoint.pt",
        "../tapnet/checkpoints/causal_bootstapir_checkpoint.pt"
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✓ Checkpoint found at: {path}")
            return path
    
    print("✗ Checkpoint not found in any expected location")
    return None

def install_tapnet_dependencies():
    """Install basic dependencies for TAPnet"""
    dependencies = [
        "torch", "torchvision", "numpy", "opencv-python", "tree"
    ]
    
    print("Installing TAPnet dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--user"])
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {dep}")

def main():
    print("TAPnet Installation Diagnostic Tool")
    print("=" * 40)
    
    # Check current installation
    tapnet_ok = check_tapnet_installation()
    checkpoint_path = check_checkpoint_exists()
    
    if tapnet_ok and checkpoint_path:
        print("\n✓ TAPnet setup appears to be working!")
        return 0
    
    print("\nIssues detected. Attempting fixes...")
    
    if not tapnet_ok:
        print("\n1. Installing TAPnet dependencies...")
        install_tapnet_dependencies()
        
        print("\n2. Checking if TAPnet directory exists...")
        tapnet_dir = "../tapnet"
        if os.path.exists(tapnet_dir):
            print(f"✓ TAPnet directory found: {tapnet_dir}")
            
            # Try to add to Python path
            import site
            user_site = site.getusersitepackages()
            os.makedirs(user_site, exist_ok=True)
            
            pth_file = os.path.join(user_site, "tapnet.pth")
            with open(pth_file, "w") as f:
                f.write(os.path.abspath(tapnet_dir))
            print(f"✓ Added TAPnet to Python path: {pth_file}")
        else:
            print(f"✗ TAPnet directory not found: {tapnet_dir}")
            print("Please clone TAPnet:")
            print("git clone https://github.com/google-deepmind/tapnet.git ../tapnet")
    
    if not checkpoint_path:
        print("\n3. Checkpoint download instructions:")
        print("Download from: https://storage.googleapis.com/dm-tapnet/causal_bootstapir_checkpoint.pt")
        print("Save to: checkpoints/causal_bootstapir_checkpoint.pt")
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        print("✓ Created checkpoints directory")
    
    print("\nAfter making the suggested changes, restart Python and try again.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
