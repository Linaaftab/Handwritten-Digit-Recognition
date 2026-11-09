# pytorch_digit_recognizer.py
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

print("🧠 HANDWRITTEN DIGIT RECOGNITION - NEURAL NETWORKS FINAL PROJECT")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    CUSTOM_DATA_DIR = r"D:\mnist code\Handwritten character recognition\custom_dataset"
    MODEL_SAVE_DIR = r"D:\HAND DIGIT RECOGNIZER NEURAL NETWORK FINAL PROJECT\models"
    RESULTS_DIR = r"D:\HAND DIGIT RECOGNIZER NEURAL NETWORK FINAL PROJECT\results"
    
    # Image parameters
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    NUM_CHANNELS = 1  # Grayscale
    NUM_CLASSES = 10  # Digits 0-9
    
    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Model parameters
    DROPOUT_RATE = 0.3

# Create directories
config = Config()
for directory in [config.MODEL_SAVE_DIR, config.RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"📁 Created directory: {directory}")

# ============================================================================
# NEURAL NETWORK ARCHITECTURES (PYTORCH)
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for digit recognition"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AdvancedCNN(nn.Module):
    """Advanced CNN with BatchNorm"""
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class DeepCNN(nn.Module):
    """Deeper CNN architecture"""
    def __init__(self):
        super(DeepCNN, self).__init__()
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_mnist_data():
    """Load MNIST dataset using PyTorch"""
    print("\n" + "="*60)
    print("📥 LOADING MNIST DATASET")
    print("="*60)
    
    try:
        # Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load MNIST
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Combine datasets
        X_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        X_test = test_dataset.data.numpy()
        y_test = test_dataset.targets.numpy()
        
        print(f"✅ MNIST Training: {X_train.shape[0]:,} images")
        print(f"✅ MNIST Test: {X_test.shape[0]:,} images")
        
        # Combine for our split
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        
        # Reshape and normalize
        X = X.astype('float32') / 255.0
        X = np.expand_dims(X, axis=1)  # Add channel dimension: (N, 1, 28, 28)
        
        # Use smaller subset for quick training
        X = X[:20000]
        y = y[:20000]
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"\n📊 Dataset Split (Subset):")
        print(f"  Training: {X_train.shape[0]:,} images")
        print(f"  Validation: {X_val.shape[0]:,} images")
        print(f"  Test: {X_test.shape[0]:,} images")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        return (X_train_tensor, y_train_tensor, 
                X_val_tensor, y_val_tensor, 
                X_test_tensor, y_test_tensor)
        
    except Exception as e:
        print(f"❌ Error loading MNIST: {e}")
        return None

def visualize_samples(X, y, title="Sample Images"):
    """Visualize sample images"""
    X_np = X.numpy() if torch.is_tensor(X) else X
    y_np = y.numpy() if torch.is_tensor(y) else y
    
    plt.figure(figsize=(12, 4))
    
    for i in range(min(10, len(X_np))):
        plt.subplot(2, 5, i+1)
        # Remove channel dimension for display
        img = X_np[i].squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f"Digit: {y_np[i]}")
        plt.axis('off')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(config.RESULTS_DIR, "training_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Visualization saved to: {save_path}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, model_name, device='cpu'):
    """Train a PyTorch model"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {model_name}")
    print(f"{'='*60}")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{config.EPOCHS}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    training_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'history': history
    }, model_path)
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    print(f"💾 Model saved to: {model_path}")
    
    return model, history

def evaluate_model(model, test_loader, model_name, device='cpu'):
    """Evaluate model performance"""
    print(f"\n📊 Evaluating {model_name}...")
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, cm, all_predictions, all_targets

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(histories):
    """Plot training history for all models"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    for model_name, history in histories.items():
        plt.plot(history['train_acc'], label=f'{model_name} Train', alpha=0.7)
        plt.plot(history['val_acc'], label=f'{model_name} Val', linestyle='--', alpha=0.7)
    plt.title('Model Accuracy During Training', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    for model_name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{model_name} Train', alpha=0.7)
        plt.plot(history['val_loss'], label=f'{model_name} Val', linestyle='--', alpha=0.7)
    plt.title('Model Loss During Training', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(config.RESULTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📈 Training history saved to: {save_path}")

def plot_confusion_matrices(results_dict):
    """Plot confusion matrices"""
    n_models = len(results_dict)
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = results['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                   xticklabels=range(10), yticklabels=range(10))
        
        accuracy = results['accuracy']
        ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f}', fontsize=12)
        ax.set_xlabel('Predicted Digit', fontsize=10)
        ax.set_ylabel('True Digit', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(config.RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Confusion matrices saved to: {save_path}")

def plot_performance_comparison(df_report):
    """Plot performance comparison bar chart"""
    plt.figure(figsize=(10, 6))
    
    # Sort by accuracy
    df_sorted = df_report.sort_values('Test Accuracy', ascending=True)
    
    # Create horizontal bar chart
    bars = plt.barh(df_sorted['Model'], df_sorted['Test Accuracy'], color='#3498db')
    plt.xlabel('Test Accuracy', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, df_sorted['Test Accuracy']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(config.RESULTS_DIR, "performance_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Performance comparison saved to: {save_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🧠 NEURAL NETWORKS FINAL PROJECT - PYTORCH VERSION")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # STEP 1: Load data
        print("\n📥 STEP 1: Loading MNIST dataset...")
        data = load_mnist_data()
        
        if data is None:
            print("❌ Failed to load data. Exiting...")
            return
        
        X_train, y_train, X_val, y_val, X_test, y_test = data
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Visualize samples
        visualize_samples(X_train[:10], y_train[:10], "Training Samples")
        
        # STEP 2: Create and train ONE model (for quick testing)
        print("\n🏗️  STEP 2: Creating and training model...")
        
        # Create model
        model = SimpleCNN()
        print("\nModel Architecture:")
        print(model)
        
        # Train model
        trained_model, history = train_model(model, train_loader, val_loader, "SimpleCNN", device)
        
        # STEP 3: Evaluate model
        print("\n📊 STEP 3: Evaluating model...")
        accuracy, cm, predictions, targets = evaluate_model(trained_model, test_loader, "SimpleCNN", device)
        
        # Store results
        histories = {"SimpleCNN": history}
        results_dict = {
            "SimpleCNN": {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'predictions': predictions,
                'targets': targets
            }
        }
        
        # STEP 4: Visualize results
        print("\n📈 STEP 4: Visualizing results...")
        plot_training_history(histories)
        plot_confusion_matrices(results_dict)
        
        # STEP 5: Create performance report
        print("\n📋 STEP 5: Creating performance report...")
        report_data = [{
            'Model': 'SimpleCNN',
            'Test Accuracy': accuracy,
            'Parameters': sum(p.numel() for p in trained_model.parameters()),
            'Model File': 'SimpleCNN.pth'
        }]
        
        df_report = pd.DataFrame(report_data)
        
        print("\n" + "="*70)
        print("📋 PERFORMANCE REPORT")
        print("="*70)
        print(df_report.to_string(index=False))
        
        # Save report to CSV
        csv_path = os.path.join(config.RESULTS_DIR, "performance_report.csv")
        df_report.to_csv(csv_path, index=False)
        print(f"\n💾 Performance report saved to: {csv_path}")
        
        # Create performance comparison chart
        plot_performance_comparison(df_report)
        
        # STEP 6: Final summary
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📈 Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"💾 Model saved in: {config.MODEL_SAVE_DIR}")
        print(f"📊 Results saved in: {config.RESULTS_DIR}")
        print("="*70)
        
        print("\n✅ Ready for your presentation!")
        print("\n📚 To test the model, run: python test_pytorch_model.py")
        
        return trained_model, accuracy, df_report
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# CREATE TEST SCRIPT
# ============================================================================

def create_test_script():
    """Create a test script for the trained model"""
    test_script = '''
# test_pytorch_model.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# Define the same model architecture
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

def test_digit_image(image_path=None):
    """Test the trained PyTorch model on a digit image"""
    
    # Load model
    model_path = r"{model_path}"
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first.")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Create model instance
    model = SimpleCNN()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # If no image provided, create a test image
    if image_path is None or not os.path.exists(image_path):
        print("Creating test digit image...")
        # Create a simple "7" digit
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(img, (20, 20), (80, 20), 255, 10)  # Top horizontal
        cv2.line(img, (80, 20), (40, 80), 255, 10)  # Diagonal
        image_path = "test_digit.png"
        cv2.imwrite(image_path, img)
        print(f"Created test image: {image_path}")
    
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("❌ Could not load image")
        return
    
    # Display
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Preprocess
    img_resized = cv2.resize(img, (28, 28))
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_resized, cmap='gray')
    plt.title('Resized (28x28)')
    plt.axis('off')
    
    # Prepare for model
    img_normalized = img_resized.astype('float32') / 255.0
    img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities).item()
        confidence = probabilities[0, predicted_digit].item()
    
    # Display prediction
    plt.subplot(1, 3, 3)
    probs_np = probabilities.numpy().flatten()
    colors = ['red' if i != predicted_digit else 'green' for i in range(10)]
    plt.bar(range(10), probs_np, color=colors)
    plt.xticks(range(10))
    plt.title(f'Prediction: {predicted_digit}\\nConfidence: {{confidence:.1%}}')
    plt.xlabel('Digit')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\\n{'='*50}")
    print("PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {{confidence:.2%}}")
    print(f"\\nConfidence scores:")
    for i, prob in enumerate(probs_np):
        star = " ⭐" if i == predicted_digit else ""
        print(f"  Digit {{i}}: {{prob:7.2%}}{{star}}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Test with your own image: test_digit_image("your_image.png")
    test_digit_image()
'''.format(model_path=os.path.join(config.MODEL_SAVE_DIR, "SimpleCNN.pth"))
    
    # Save test script
    test_path = os.path.join(os.path.dirname(__file__), "test_pytorch_model.py")
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    print(f"\n📝 Test script created: test_pytorch_model.py")
    print("Run it with: python test_pytorch_model.py")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("Starting PyTorch Neural Networks Project...")
    
    # Run main project
    model = main()
    
    # Create test script
    if model is not None:
        create_test_script()