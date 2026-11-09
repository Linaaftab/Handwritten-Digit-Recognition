# check_files.py
import os

print("="*60)
print("CHECKING FILES IN YOUR PROJECT DIRECTORY")
print("="*60)

print(f"\nCurrent directory: {os.getcwd()}")

print("\n📁 All files and folders:")
print("-" * 40)

all_files = os.listdir('.')
for i, file in enumerate(all_files, 1):
    file_size = os.path.getsize(file) if os.path.isfile(file) else "DIR"
    print(f"{i:3}. {file:40} ({file_size})")

print("\n📄 Python files (.py):")
print("-" * 40)
py_files = [f for f in all_files if f.endswith('.py')]
for file in py_files:
    print(f"  - {file}")

print("\n📊 Data files:")
print("-" * 40)
data_files = [f for f in all_files if f.endswith(('.csv', '.png', '.jpg', '.txt'))]
for file in data_files:
    print(f"  - {file}")

print("\n🔍 Looking for model checkpoint patterns...")
model_patterns = ['.pth', '.pt', '.bin', 'model', 'checkpoint']
for file in all_files:
    for pattern in model_patterns:
        if pattern in file.lower():
            print(f"  ⚠️ Possible model file: {file}")