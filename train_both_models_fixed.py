# train_both_models_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

print("="*70)
print("TRAINING MODELS ON MNIST AND CUSTOM DATASETS")
print("="*70)

# Define the CNN model
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

class CustomDigitDataset(Dataset):
    """Load your custom handwritten digits"""
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        
        print(f"Loading custom dataset from: {data_dir}")
        
        for digit in range(10):
            digit_dir = os.path.join(data_dir, str(digit))
            if os.path.exists(digit_dir):
                image_files = [f for f in os.listdir(digit_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                for img_file in tqdm(image_files, desc=f"Digit {digit}", leave=False):
                    img_path = os.path.join(digit_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize to 28x28
                        img = cv2.resize(img, (28, 28))
                        
                        # Convert to float and normalize
                        img = img.astype('float32') / 255.0
                        
                        # Normalize like MNIST
                        img = (img - 0.1307) / 0.3081
                        
                        self.images.append(img)
                        self.labels.append(digit)
        
        print(f"Loaded {len(self.images)} images from custom dataset")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_tensor = torch.FloatTensor(self.images[idx]).unsqueeze(0)
        label_tensor = torch.LongTensor([self.labels[idx]])[0]
        return img_tensor, label_tensor

def train_model(model, train_loader, val_loader, test_loader, dataset_name, epochs=10):
    """Train a model and return accuracy"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🚀 Training on {dataset_name} dataset...")
    
    train_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Test the best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    
    print(f'\n📊 {dataset_name} Test Accuracy: {test_acc:.2f}% ({test_correct}/{test_total})')
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_accs, dataset_name)
    
    return model, test_acc, train_losses, train_accs, val_accs

def plot_training_history(train_losses, train_accs, val_accs, dataset_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{dataset_name} - Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{dataset_name} - Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name.lower()}_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_models_and_results(mnist_model, mnist_acc, custom_model, custom_acc, custom_dataset_size):
    """Save models and results"""
    print("\n" + "="*70)
    print("SAVING MODELS AND RESULTS")
    print("="*70)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save MNIST model
    mnist_path = 'models/mnist_model.pth'
    torch.save(mnist_model.state_dict(), mnist_path)
    print(f"✅ MNIST model saved to: {mnist_path}")
    print(f"   Accuracy: {mnist_acc:.2f}%")
    
    # Save Custom model
    custom_path = 'models/custom_model.pth'
    torch.save(custom_model.state_dict(), custom_path)
    print(f"✅ Custom model saved to: {custom_path}")
    print(f"   Accuracy: {custom_acc:.2f}%")
    
    # Create comparison results
    results = {
        'Model': ['MNIST Model', 'Custom Model'],
        'Dataset': ['MNIST (70,000 samples)', f'Custom ({custom_dataset_size} samples)'],
        'Test Accuracy (%)': [mnist_acc, custom_acc],
        'Parameters': [sum(p.numel() for p in mnist_model.parameters()), 
                      sum(p.numel() for p in custom_model.parameters())],
        'Model File': ['models/mnist_model.pth', 'models/custom_model.pth']
    }
    
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    df_results.to_csv('model_comparison.csv', index=False)
    print(f"✅ Model comparison saved to: model_comparison.csv")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(df_results['Model'], df_results['Test Accuracy (%)'], 
                  color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: MNIST vs Custom Dataset', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, df_results['Test Accuracy (%)']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_chart.png', dpi=150, bbox_inches='tight')
    print(f"✅ Comparison chart saved to: model_comparison_chart.png")
    
    # Create detailed model info file
    with open('models/model_info.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL INFORMATION - HANDWRITTEN DIGIT RECOGNITION\n")
        f.write("="*60 + "\n\n")
        
        f.write("MODEL ARCHITECTURE:\n")
        f.write("-"*40 + "\n")
        f.write("SimpleCNN:\n")
        f.write("  Conv2d(1, 32, kernel_size=3, padding=1) → ReLU → MaxPool2d\n")
        f.write("  Conv2d(32, 64, kernel_size=3, padding=1) → ReLU → MaxPool2d\n")
        f.write("  Flatten\n")
        f.write("  Linear(64*7*7, 128) → ReLU → Dropout(0.3)\n")
        f.write("  Linear(128, 10) → Softmax\n")
        f.write(f"  Total Parameters: {sum(p.numel() for p in mnist_model.parameters()):,}\n\n")
        
        f.write("TRAINING DETAILS:\n")
        f.write("-"*40 + "\n")
        f.write("Optimizer: Adam (lr=0.001)\n")
        f.write("Loss Function: CrossEntropyLoss\n")
        f.write("Batch Size: 64 (MNIST), 32 (Custom)\n")
        f.write("Epochs: 10 (MNIST), 15 (Custom)\n")
        f.write("Validation Split: 20% (MNIST), 15% (Custom)\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-"*40 + "\n")
        f.write(f"MNIST Model: {mnist_acc:.2f}% accuracy\n")
        f.write(f"Custom Model: {custom_acc:.2f}% accuracy\n")
        f.write(f"Accuracy Gap: {abs(mnist_acc - custom_acc):.2f}%\n\n")
        
        f.write("FILES:\n")
        f.write("-"*40 + "\n")
        f.write("mnist_model.pth - Trained on MNIST dataset\n")
        f.write("custom_model.pth - Trained on custom dataset\n")
        f.write("model_comparison.csv - Performance comparison\n")
        f.write("model_comparison_chart.png - Visualization\n")
        
    print(f"✅ Model info saved to: models/model_info.txt")

def main():
    """Main training function"""
    
    # Step 1: Train MNIST model
    print("\n" + "="*70)
    print("STEP 1: TRAINING ON MNIST DATASET")
    print("="*70)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split train into train and validation
    train_size = int(0.8 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, val_size])
    
    print(f"MNIST Dataset:")
    print(f"  Training samples: {len(mnist_train)}")
    print(f"  Validation samples: {len(mnist_val)}")
    print(f"  Test samples: {len(mnist_test)}")
    
    # Create data loaders
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(mnist_val, batch_size=64, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
    
    # Create and train MNIST model
    mnist_model = SimpleCNN()
    mnist_model, mnist_acc, mnist_losses, mnist_train_accs, mnist_val_accs = train_model(
        mnist_model, train_loader, val_loader, test_loader, "MNIST", epochs=10
    )
    
    # Step 2: Train Custom model
    print("\n" + "="*70)
    print("STEP 2: TRAINING ON CUSTOM DATASET")
    print("="*70)
    
    # Load custom dataset
    custom_data_dir = r"D:\HAND DIGIT RECOGNIZER NEURAL NETWORK FINAL PROJECT\custom_dataset"
    
    if not os.path.exists(custom_data_dir):
        print(f"❌ Custom dataset directory not found: {custom_data_dir}")
        print("Please check the path and try again.")
        return
    
    custom_dataset = CustomDigitDataset(custom_data_dir)
    custom_dataset_size = len(custom_dataset)
    
    if custom_dataset_size == 0:
        print("❌ No images found in custom dataset")
        return
    
    # Split custom data
    custom_train_size = int(0.7 * custom_dataset_size)
    custom_val_size = int(0.15 * custom_dataset_size)
    custom_test_size = custom_dataset_size - custom_train_size - custom_val_size
    
    custom_train, custom_val, custom_test = torch.utils.data.random_split(
        custom_dataset, [custom_train_size, custom_val_size, custom_test_size]
    )
    
    print(f"\nCustom Dataset:")
    print(f"  Training samples: {len(custom_train)}")
    print(f"  Validation samples: {len(custom_val)}")
    print(f"  Test samples: {len(custom_test)}")
    
    # Create data loaders for custom data
    custom_train_loader = DataLoader(custom_train, batch_size=32, shuffle=True)
    custom_val_loader = DataLoader(custom_val, batch_size=32, shuffle=False)
    custom_test_loader = DataLoader(custom_test, batch_size=32, shuffle=False)
    
    # Create and train Custom model
    custom_model = SimpleCNN()
    custom_model, custom_acc, custom_losses, custom_train_accs, custom_val_accs = train_model(
        custom_model, custom_train_loader, custom_val_loader, custom_test_loader, 
        "Custom", epochs=15  # More epochs for smaller dataset
    )
    
    # Step 3: Save everything
    save_models_and_results(mnist_model, mnist_acc, custom_model, custom_acc, custom_dataset_size)
    
    # Step 4: Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE! 🎉")
    print("="*70)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  MNIST Model Accuracy: {mnist_acc:.2f}%")
    print(f"  Custom Model Accuracy: {custom_acc:.2f}%")
    print(f"  Accuracy Difference: {abs(mnist_acc - custom_acc):.2f}%")
    
    print(f"\n📁 FILES CREATED:")
    print(f"  models/mnist_model.pth")
    print(f"  models/custom_model.pth")
    print(f"  model_comparison.csv")
    print(f"  model_comparison_chart.png")
    print(f"  mnist_training_history.png")
    print(f"  custom_training_history.png")
    print(f"  models/model_info.txt")
    
    print(f"\n🎯 NEXT STEP:")
    print(f"  Run the GUI: python dual_model_gui.py")
    
    print(f"\n💡 INSIGHTS:")
    if custom_acc < 70:
        print(f"  ⚠️  Custom model accuracy is low ({custom_acc:.2f}%).")
        print(f"  Recommendations:")
        print(f"    1. Add more images to your custom dataset")
        print(f"    2. Ensure consistent image quality")
        print(f"    3. Try data augmentation")
    elif abs(mnist_acc - custom_acc) > 20:
        print(f"  ⚠️  Large accuracy gap ({abs(mnist_acc - custom_acc):.2f}%).")
        print(f"  This suggests significant differences between datasets.")
    else:
        print(f"  ✅ Good performance! Models generalize well.")

if __name__ == "__main__":
    main()