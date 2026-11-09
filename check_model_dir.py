# check_models_dir.py
import os

print("="*60)
print("CHECKING MODELS DIRECTORY")
print("="*60)

models_dir = 'models'
if os.path.exists(models_dir):
    print(f"✅ Models directory found: {models_dir}")
    
    print("\n📁 Contents of models directory:")
    print("-" * 40)
    
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  - {file:30} ({size:,} bytes)")
        else:
            print(f"  - {file:30} (DIR)")
    
    # Look for model files specifically
    print("\n🔍 Looking for model files (.pth, .pt, .bin):")
    print("-" * 40)
    
    model_extensions = ['.pth', '.pt', '.bin', '.ckpt']
    found_models = []
    
    for file in os.listdir(models_dir):
        if any(file.endswith(ext) for ext in model_extensions):
            file_path = os.path.join(models_dir, file)
            size = os.path.getsize(file_path)
            found_models.append((file, size))
            print(f"  ✅ Found: {file} ({size:,} bytes)")
    
    if not found_models:
        print("  ❌ No model files found in models directory")
        
else:
    print(f"❌ Models directory not found: {models_dir}")

# Also check if there are model files in the root directory
print("\n🔍 Checking root directory for model files:")
print("-" * 40)

root_model_files = []
for file in os.listdir('.'):
    if any(file.endswith(ext) for ext in model_extensions):
        size = os.path.getsize(file)
        root_model_files.append((file, size))
        print(f"  ✅ Found: {file} ({size:,} bytes)")

if not root_model_files:
    print("  ❌ No model files in root directory")