#!/usr/bin/env python3
"""
BabyBIONN Startup Script - ENHANCED VERSION
Run from project root: python run_babybionn.py
"""

import subprocess
import sys
import os
import time
import spacy

def load_spacy_model():
    """Self-healing spaCy model loader"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("📥 Downloading spaCy model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")
    
def verify_enhanced_dependencies():
    """Verify all enhanced dependencies are working"""
    print("🔍 Verifying enhanced dependencies...")
    dependencies_ok = True
    
    try:
        nlp = load_spacy_model()  # ← CHANGED: Use the self-healing function here
        print("✅ spaCy loaded successfully")
    except Exception as e:
        print(f"❌ spaCy failed: {e}")
        dependencies_ok = False

    #try:
    #    from ultralytics import YOLO
    #    model = YOLO('yolov8n.pt')
    #    print("✅ YOLO loaded successfully")
    #except Exception as e:
    #    print(f"❌ YOLO failed: {e}")
    #    dependencies_ok = False

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✅ PyTorch loaded (CUDA: {cuda_available})")
        if cuda_available:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"❌ PyTorch failed: {e}")
        dependencies_ok = False

    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        print("✅ Transformers loaded successfully")
    except Exception as e:
        print(f"❌ Transformers failed: {e}")
        dependencies_ok = False

    try:
        from enhanced_vni_classes import EnhancedMedicalVNI
        medical_vni = EnhancedMedicalVNI("test_instance")
        print("✅ Enhanced VNI classes loaded successfully")
    except Exception as e:
        print(f"❌ Enhanced VNI classes failed: {e}")
        dependencies_ok = False

    return dependencies_ok

def main():
    """Start the BabyBIONN system"""
    print("🚀 Starting BabyBIONN AI System - ENHANCED VERSION...")
    print("=" * 60)
    
    # Check if we're in the right directory
    main_py_path = os.path.join(os.getcwd(), "main.py")
    if not os.path.exists(main_py_path):
        print("❌ 'main.py' not found. Please run from project root.")
        sys.exit(1)
    
    # Verify dependencies before starting
    print("\n📦 Checking enhanced dependencies...")
    deps_ok = verify_enhanced_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Some enhanced dependencies failed, but starting anyway...")
        print("   Basic functionality may work, but enhanced features might be limited.")
    else:
        print("\n🎉 All enhanced dependencies verified successfully!")
    
    print("\n🌐 Starting BabyBIONN server...")
    print("   Server will be available at: http://localhost:8001")
    print("   API docs at: http://localhost:8001/docs")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the FastAPI server from the neuron module
        subprocess.run([
            sys.executable, 
            "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "8001", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 BabyBIONN system stopped.")
    except Exception as e:
        print(f"❌ Error starting BabyBIONN: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   - Make sure port 8001 is available")
        print("   - Check if all dependencies are installed")
        print("   - Verify you're in the correct directory")

if __name__ == "__main__":
    main()
