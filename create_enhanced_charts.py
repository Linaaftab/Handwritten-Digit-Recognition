# create_enhanced_charts.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Your results
data = {
    'Dataset': ['MNIST', 'Custom Dataset'],
    'Test Accuracy (%)': [99.23, 73.33],
    'Training Samples': [48000, 1400],
    'Accuracy Gap': [0, 25.9]
}

df = pd.DataFrame(data)

# Create multi-panel visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MNIST vs Custom Dataset: Comprehensive Analysis', fontsize=16, fontweight='bold')

# 1. Accuracy Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(df['Dataset'], df['Test Accuracy (%)'], 
                color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars1, df['Test Accuracy (%)']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 2. Dataset Size Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(df['Dataset'], df['Training Samples'], 
                color=['#2ecc71', '#f39c12'], edgecolor='black', linewidth=2)
ax2.set_ylabel('Training Samples', fontsize=12, fontweight='bold')
ax2.set_title('Dataset Size Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, samples in zip(bars2, df['Training Samples']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1000,
            f'{samples:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# 3. Training Progress (simulated)
ax3 = axes[1, 0]
epochs = list(range(1, 6))
mnist_acc = [94.87, 98.10, 98.60, 98.88, 99.13]
custom_acc = [14.57, 28.93, 47.14, 59.00, 66.86]

ax3.plot(epochs, mnist_acc, 'b-o', linewidth=2, markersize=8, label='MNIST')
ax3.plot(epochs, custom_acc, 'r-s', linewidth=2, markersize=8, label='Custom')
ax3.set_xlabel('Epochs', fontsize=12)
ax3.set_ylabel('Training Accuracy (%)', fontsize=12)
ax3.set_title('Training Progress Comparison', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(epochs)

# 4. Gap Analysis
ax4 = axes[1, 1]
gap_data = ['MNIST Performance', 'Accuracy Gap', 'Custom Performance']
gap_values = [99.23, 25.9, 73.33]
colors = ['#3498db', '#9b59b6', '#e74c3c']

bars4 = ax4.bar(gap_data, gap_values, color=colors, edgecolor='black', linewidth=2)
ax4.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax4.set_title('25.9% Accuracy Gap Analysis', fontsize=14, fontweight='bold')
ax4.set_ylim(0, 110)
ax4.grid(axis='y', alpha=0.3)

for bar, value in zip(bars4, gap_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{value}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('enhanced_comparison_charts.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Enhanced charts saved as 'enhanced_comparison_charts.png'")