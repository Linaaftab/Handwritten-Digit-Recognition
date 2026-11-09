# find_and_rename_model.py
import os
import torch

print("Looking for your trained model files...")

# Common model file names to check
possible_names = [
    'model.pth', 'model_mnist.pth', 'mnist_model.pth', 
    'trained_model.pth', 'cnn_model.pth', 'digit_model.pth',
    'best_model.pth', 'final_model.pth'
]

found_models = []

for filename in possible_names:
    if os.path.exists(filename):
        print(f"✅ Found: {filename}")
        found_models.append(filename)

if found_models:
    print(f"\nFound {len(found_models)} model file(s).")
    
    # Try to load each one
    for model_file in found_models:
        try:
            # Try to load the model
            model_dict = torch.load(model_file, map_location='cpu')
            print(f"\n📦 Model info from '{model_file}':")
            print(f"   Number of parameter groups: {len(model_dict)}")
            
            # Estimate total parameters
            total_params = sum(p.numel() for p in model_dict.values() if torch.is_tensor(p))
            print(f"   Approx. total parameters: {total_params:,}")
            
            # Ask if this is the MNIST model
            response = input(f"\nIs '{model_file}' your MNIST model? (y/n): ").lower()
            
            if response == 'y':
                # Rename to standard name for GUI
                torch.save(model_dict, 'mnist_model.pth')
                print(f"✅ Renamed to 'mnist_model.pth'")
                break
                
        except Exception as e:
            print(f"   Could not load {model_file}: {e}")
    
    print("\n🎯 Now run: python digit_recognizer_gui.py")
    
else:
    print("❌ No model files found in current directory.")
    print("\n📁 Files in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.pth') or file.endswith('.pt'):
            print(f"  - {file}")