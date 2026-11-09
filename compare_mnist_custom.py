# compare_mnist_custom.py
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("HANDWRITTEN DIGIT RECOGNITION: MNIST vs CUSTOM DATASET COMPARISON")
print("="*70)

# Check if custom dataset exists
CUSTOM_DATA_DIR = r"D:\HAND DIGIT RECOGNIZER NEURAL NETWORK FINAL PROJECT\custom_dataset"
if not os.path.exists(CUSTOM_DATA_DIR):
    print(f"❌ ERROR: Custom dataset directory not found!")
    print(f"Expected path: {CUSTOM_DATA_DIR}")
    print("\n📝 Please create the directory structure:")
    print("custom_dataset/")
    print("├── 0/   (put images of digit 0 here)")
    print("├── 1/   (put images of digit 1 here)")
    print("├── ...")
    print("└── 9/   (put images of digit 9 here)")
    sys.exit(1)

class CustomDigitDataset(Dataset):
    """Load your custom handwritten digits"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        print(f"\n📂 Loading custom dataset from: {data_dir}")
        
        for digit in range(10):
            digit_dir = os.path.join(data_dir, str(digit))
            if os.path.exists(digit_dir):
                image_files = [f for f in os.listdir(digit_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                if not image_files:
                    print(f"⚠️  Warning: No images found in {digit_dir}")
                    continue
                    
                for img_file in tqdm(image_files, desc=f"Digit {digit}", leave=False):
                    img_path = os.path.join(digit_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        self.images.append(img)
                        self.labels.append(digit)
                    else:
                        print(f"⚠️  Could not read image: {img_path}")
        
        print(f"✅ Loaded {len(self.images)} images from custom dataset")
        
        if len(self.images) == 0:
            print("❌ ERROR: No images loaded! Check your dataset structure.")
            sys.exit(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Resize to 28x28 (MNIST size)
        img = cv2.resize(img, (28, 28))
        img = img.astype('float32') / 255.0
        
        # Normalize like MNIST
        img = (img - 0.1307) / 0.3081
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension
        label_tensor = torch.LongTensor([label])[0]
        
        return img_tensor, label_tensor

# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, epochs=5, dataset_name="Dataset"):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🚀 Training on {dataset_name}...")
    
    train_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_val_acc': val_accs[-1]
    }

def test_model(model, test_loader, dataset_name="Test"):
    """Test the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Testing on {dataset_name}'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    print(f'\n📊 {dataset_name} Results:')
    print(f'Accuracy: {accuracy:.2f}% ({correct}/{total})')
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print(f'\n📈 Confusion Matrix Generated')
    
    return accuracy, cm

def load_mnist():
    """Load MNIST dataset"""
    print("\n" + "="*50)
    print("LOADING MNIST DATASET")
    print("="*50)
    
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Downloading MNIST dataset...")
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split train into train and validation
    train_size = int(0.8 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = torch.utils.data.random_split(
        mnist_train, [train_size, val_size]
    )
    
    print(f"✅ MNIST loaded successfully!")
    print(f"   Training set: {len(mnist_train)} images")
    print(f"   Validation set: {len(mnist_val)} images")
    print(f"   Test set: {len(mnist_test)} images")
    
    return mnist_train, mnist_val, mnist_test

def train_and_compare():
    """Main function to train and compare on both datasets"""
    
    # Load MNIST
    mnist_train, mnist_val, mnist_test = load_mnist()
    
    # Load Custom Dataset
    print("\n" + "="*50)
    print("LOADING CUSTOM DATASET")
    print("="*50)
    
    custom_dataset = CustomDigitDataset(CUSTOM_DATA_DIR)
    
    # Split custom data
    custom_train_size = int(0.7 * len(custom_dataset))
    custom_val_size = int(0.15 * len(custom_dataset))
    custom_test_size = len(custom_dataset) - custom_train_size - custom_val_size
    
    custom_train, custom_val, custom_test = torch.utils.data.random_split(
        custom_dataset, [custom_train_size, custom_val_size, custom_test_size]
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"{'Dataset':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 50)
    print(f"{'MNIST':<15} {len(mnist_train):<10} {len(mnist_val):<10} {len(mnist_test):<10} {len(mnist_train)+len(mnist_val)+len(mnist_test):<10}")
    print(f"{'Custom':<15} {len(custom_train):<10} {len(custom_val):<10} {len(custom_test):<10} {len(custom_dataset):<10}")
    
    # Create DataLoaders
    batch_size = 32
    
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    
    custom_train_loader = DataLoader(custom_train, batch_size=batch_size, shuffle=True)
    custom_val_loader = DataLoader(custom_val, batch_size=batch_size, shuffle=False)
    custom_test_loader = DataLoader(custom_test, batch_size=batch_size, shuffle=False)
    
    # Train on MNIST
    print("\n" + "="*60)
    print("TRAINING MODEL ON MNIST DATASET")
    print("="*60)
    
    model_mnist = SimpleCNN()
    mnist_results = train_model(
        model_mnist, 
        mnist_train_loader, 
        mnist_val_loader, 
        epochs=5, 
        dataset_name="MNIST"
    )
    
    # Test on MNIST
    mnist_acc, mnist_cm = test_model(model_mnist, mnist_test_loader, "MNIST Test")
    
    # Train on Custom Data
    print("\n" + "="*60)
    print("TRAINING MODEL ON CUSTOM DATASET")
    print("="*60)
    
    model_custom = SimpleCNN()
    custom_results = train_model(
        model_custom, 
        custom_train_loader, 
        custom_val_loader, 
        epochs=5, 
        dataset_name="Custom Dataset"
    )
    
    # Test on Custom Data
    custom_acc, custom_cm = test_model(model_custom, custom_test_loader, "Custom Dataset Test")
    
    # Create comparison table
    results_data = {
        'Dataset': ['MNIST', 'Custom'],
        'Training Samples': [len(mnist_train), len(custom_train)],
        'Test Accuracy (%)': [mnist_acc, custom_acc],
        'Model Parameters': [sum(p.numel() for p in model_mnist.parameters()), 
                           sum(p.numel() for p in model_custom.parameters())]
    }
    
    df_results = pd.DataFrame(results_data)
    
    print("\n" + "="*70)
    print("DATASET COMPARISON RESULTS")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # Create visualization
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Accuracy Comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(df_results['Dataset'], df_results['Test Accuracy (%)'], 
                   color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=2)
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Performance: MNIST vs Custom Dataset', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, df_results['Test Accuracy (%)']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Training Samples Comparison
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(df_results['Dataset'], df_results['Training Samples'],
                    color=['#9b59b6', '#e74c3c'], edgecolor='black', linewidth=2)
    plt.ylabel('Number of Training Samples', fontsize=12, fontweight='bold')
    plt.title('Dataset Size Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, samples in zip(bars2, df_results['Training Samples']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{samples:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results to CSV
    df_results.to_csv('comparison_results.csv', index=False)
    print(f"\n💾 Results saved to 'comparison_results.csv'")
    print(f"📊 Visualization saved to 'dataset_comparison.png'")
    
    # Additional insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    if len(custom_train) < 1000:
        print("⚠️  Custom dataset is relatively small. Consider:")
        print("   - Collecting more samples (100+ per digit recommended)")
        print("   - Using data augmentation techniques")
        print("   - Trying transfer learning (pre-train on MNIST, fine-tune on custom)")
    
    accuracy_difference = abs(mnist_acc - custom_acc)
    if accuracy_difference > 20:
        print(f"⚠️  Large accuracy gap ({accuracy_difference:.2f}%) detected!")
        print("   This suggests significant differences between MNIST and custom data.")
        print("   Possible reasons:")
        print("   - Different writing styles")
        print("   - Image quality variations")
        print("   - Background/foreground differences")
    
    return df_results

def quick_test_with_existing_model():
    """Quick test using the pre-trained model from your project"""
    print("\n⚡ QUICK TEST: Using your pre-trained model on custom data")
    
    # Load your pre-trained model
    try:
        from test_model import load_model
        model = load_model()
        print("✅ Pre-trained model loaded successfully!")
        
        # Load custom dataset
        custom_dataset = CustomDigitDataset(CUSTOM_DATA_DIR)
        custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=False)
        
        # Test the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(custom_loader, desc="Testing pre-trained model"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        print(f"\n📊 Pre-trained Model Performance on Custom Data:")
        print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        if accuracy < 70:
            print("\n🔧 Recommendation: The pre-trained model doesn't generalize well.")
            print("   You should train a new model specifically on your custom data.")
        
        return accuracy
        
    except ImportError:
        print("⚠️  Could not import pre-trained model. Run full comparison instead.")
        return None

if __name__ == "__main__":
    print("\n🎯 HANDWRITTEN DIGIT RECOGNITION PROJECT")
    print("🎯 MNIST vs CUSTOM DATASET COMPARISON")
    print("\nOptions:")
    print("1. Run full comparison (train on both datasets)")
    print("2. Quick test (use pre-trained model on custom data)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        print("\n" + "="*70)
        print("STARTING FULL COMPARISON")
        print("="*70)
        results = train_and_compare()
    elif choice == '2':
        results = quick_test_with_existing_model()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice. Running full comparison...")
        results = train_and_compare()
    
    print("\n" + "="*70)
    print("PROGRAM COMPLETED SUCCESSFULLY!")
    print("="*70)