# dual_model_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pandas as pd

print("="*60)
print("DUAL MODEL DIGIT RECOGNIZER GUI")
print("="*60)

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

class DualModelDigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Model Digit Recognizer")
        self.root.geometry("1200x700")
        
        # Set window properties
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.image_path = None
        self.current_image = None
        
        # Your actual accuracy results
        self.mnist_acc = 99.03
        self.custom_acc = 86.67
        
        # Load both models
        self.mnist_model = self.load_model('mnist')
        self.custom_model = self.load_model('custom')
        
        # Create GUI
        self.create_widgets()
        
        # Show welcome
        self.show_welcome_message()
    
    def load_model(self, model_type):
        """Load a trained model"""
        model_path = f'models/{model_type}_model.pth'
        
        if not os.path.exists(model_path):
            print(f"⚠️ {model_type.capitalize()} model not found: {model_path}")
            print("   Using default model")
            model = SimpleCNN()
            return model
        
        try:
            model = SimpleCNN()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            print(f"✅ {model_type.capitalize()} model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Error loading {model_type} model: {e}")
            model = SimpleCNN()
            return model
    
    def show_welcome_message(self):
        """Show welcome message"""
        welcome_text = f"""
        🎉 DUAL MODEL DIGIT RECOGNIZER 🎉
        
        This application uses two neural network models:
        
        1. MNIST Model: Trained on standard MNIST dataset
           Accuracy: {self.mnist_acc}%
        
        2. Custom Model: Trained on your handwritten digits
           Accuracy: {self.custom_acc}%
        
        Features:
        • Upload images or draw digits
        • Compare both models simultaneously
        • View detailed probability distributions
        • Export recognition results
        
        🚀 Ready to test both models!
        """
        
        messagebox.showinfo("Welcome to Dual Model Digit Recognizer", welcome_text)
    
    def create_widgets(self):
        """Create the GUI layout"""
        
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#34495e', height=120)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(title_frame,
                              text="✏️ DUAL MODEL DIGIT RECOGNIZER",
                              font=('Arial', 24, 'bold'),
                              bg='#34495e',
                              fg='#ecf0f1')
        title_label.pack(pady=5)
        
        # Subtitle
        subtitle_text = f"Neural Network Project | Compare MNIST vs Custom Models"
        subtitle_label = tk.Label(title_frame,
                                 text=subtitle_text,
                                 font=('Arial', 12),
                                 bg='#34495e',
                                 fg='#bdc3c7')
        subtitle_label.pack()
        
        # Model status
        status_frame = tk.Frame(title_frame, bg='#34495e')
        status_frame.pack(pady=10)
        
        mnist_status = tk.Label(status_frame,
                               text=f"MNIST Model: {self.mnist_acc}% accuracy",
                               font=('Arial', 10, 'bold'),
                               bg='#3498db',
                               fg='white',
                               padx=10,
                               pady=2)
        mnist_status.pack(side=tk.LEFT, padx=5)
        
        custom_status = tk.Label(status_frame,
                                text=f"Custom Model: {self.custom_acc}% accuracy",
                                font=('Arial', 10, 'bold'),
                                bg='#2ecc71',
                                fg='white',
                                padx=10,
                                pady=2)
        custom_status.pack(side=tk.LEFT, padx=5)
        
        # Main Content Frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left Panel - Input
        left_panel = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)
        
        # Right Panel - Results
        right_panel = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        
        # Setup panels
        self.setup_input_panel(left_panel)
        self.setup_results_panel(right_panel)
    
    def setup_input_panel(self, parent):
        """Setup input panel"""
        
        # Panel title
        input_title = tk.Label(parent,
                              text="📤 INPUT DIGIT",
                              font=('Arial', 14, 'bold'),
                              bg='#3498db',
                              fg='white')
        input_title.pack(fill=tk.X, pady=(0, 10))
        
        # Image display
        image_frame = tk.Frame(parent, bg='white', height=300)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.image_label = tk.Label(image_frame,
                                   text="No Image Selected\n\nUse buttons below to input a digit",
                                   font=('Arial', 12),
                                   bg='white',
                                   fg='#7f8c8d')
        self.image_label.pack(expand=True, padx=20, pady=20)
        
        # Input buttons
        btn_frame = tk.Frame(parent, bg='#ecf0f1')
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Row 1
        row1 = tk.Frame(btn_frame, bg='#ecf0f1')
        row1.pack()
        
        upload_btn = tk.Button(row1,
                              text="📁 Upload Image",
                              command=self.upload_image,
                              font=('Arial', 11),
                              bg='#3498db',
                              fg='white',
                              width=20,
                              height=2)
        upload_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        draw_btn = tk.Button(row1,
                            text="✏️ Draw Digit",
                            command=self.open_drawing_window,
                            font=('Arial', 11),
                            bg='#2ecc71',
                            fg='white',
                            width=20,
                            height=2)
        draw_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Row 2
        row2 = tk.Frame(btn_frame, bg='#ecf0f1')
        row2.pack()
        
        clear_btn = tk.Button(row2,
                             text="🗑️ Clear",
                             command=self.clear_image,
                             font=('Arial', 11),
                             bg='#e74c3c',
                             fg='white',
                             width=20,
                             height=2)
        clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        samples_btn = tk.Button(row2,
                               text="🎲 Samples",
                               command=self.show_samples,
                               font=('Arial', 11),
                               bg='#9b59b6',
                               fg='white',
                               width=20,
                               height=2)
        samples_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Sample digits
        sample_frame = tk.LabelFrame(parent,
                                    text="Quick Test Digits",
                                    font=('Arial', 12, 'bold'),
                                    bg='#ecf0f1',
                                    padx=10,
                                    pady=10)
        sample_frame.pack(fill=tk.X, padx=20, pady=10)
        
        sample_btn_frame = tk.Frame(sample_frame, bg='#ecf0f1')
        sample_btn_frame.pack()
        
        for i in range(10):
            btn = tk.Button(sample_btn_frame,
                          text=str(i),
                          command=lambda d=i: self.create_sample_digit(d),
                          font=('Arial', 12, 'bold'),
                          bg='#f39c12',
                          fg='white',
                          width=4,
                          height=2)
            btn.grid(row=i//5, column=i%5, padx=2, pady=2)
        
        # Recognize button
        recognize_frame = tk.Frame(parent, bg='#ecf0f1')
        recognize_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.recognize_btn = tk.Button(recognize_frame,
                                      text="🔍 COMPARE BOTH MODELS",
                                      command=self.compare_both_models,
                                      font=('Arial', 14, 'bold'),
                                      bg='#e74c3c',
                                      fg='white',
                                      width=30,
                                      height=2,
                                      state=tk.DISABLED)
        self.recognize_btn.pack()
    
    def setup_results_panel(self, parent):
        """Setup results panel"""
        
        # Panel title
        results_title = tk.Label(parent,
                                text="📊 COMPARISON RESULTS",
                                font=('Arial', 14, 'bold'),
                                bg='#9b59b6',
                                fg='white')
        results_title.pack(fill=tk.X, pady=(0, 10))
        
        # Create notebook for tabs
        self.results_notebook = ttk.Notebook(parent)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.mnist_tab = tk.Frame(self.results_notebook, bg='white')
        self.custom_tab = tk.Frame(self.results_notebook, bg='white')
        self.comparison_tab = tk.Frame(self.results_notebook, bg='white')
        
        self.results_notebook.add(self.mnist_tab, text=f"MNIST Model ({self.mnist_acc}%)")
        self.results_notebook.add(self.custom_tab, text=f"Custom Model ({self.custom_acc}%)")
        self.results_notebook.add(self.comparison_tab, text="Comparison")
        
        # Initialize empty results
        self.clear_all_results()
    
    def clear_all_results(self):
        """Clear all result tabs"""
        for widget in self.mnist_tab.winfo_children():
            widget.destroy()
        for widget in self.custom_tab.winfo_children():
            widget.destroy()
        for widget in self.comparison_tab.winfo_children():
            widget.destroy()
        
        # Add placeholder messages
        placeholder_text = "Results will appear here after recognition"
        
        for tab in [self.mnist_tab, self.custom_tab, self.comparison_tab]:
            label = tk.Label(tab, text=placeholder_text, 
                           font=('Arial', 14), bg='white', fg='#95a5a6')
            label.pack(expand=True)
    
    def upload_image(self):
        """Upload an image"""
        file_path = filedialog.askopenfilename(
            title="Select a digit image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if file_path:
            self.display_image(file_path)
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            img = Image.open(image_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            self.current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.image_path = image_path
            
            self.recognize_btn.config(state=tk.NORMAL, bg='#2ecc71')
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def open_drawing_window(self):
        """Open drawing window"""
        drawing_window = tk.Toplevel(self.root)
        drawing_window.title("Draw a Digit")
        drawing_window.geometry("400x500")
        
        canvas = tk.Canvas(drawing_window, width=280, height=280, bg='white', cursor='cross')
        canvas.pack(pady=20)
        
        last_x, last_y = None, None
        
        def start_draw(event):
            nonlocal last_x, last_y
            last_x, last_y = event.x, event.y
        
        def draw(event):
            nonlocal last_x, last_y
            if last_x and last_y:
                canvas.create_line(last_x, last_y, event.x, event.y,
                                 width=15, fill='black', capstyle=tk.ROUND, smooth=True)
            last_x, last_y = event.x, event.y
        
        def stop_draw(event):
            nonlocal last_x, last_y
            last_x, last_y = None, None
        
        canvas.bind("<Button-1>", start_draw)
        canvas.bind("<B1-Motion>", draw)
        canvas.bind("<ButtonRelease-1>", stop_draw)
        
        btn_frame = tk.Frame(drawing_window)
        btn_frame.pack(pady=10)
        
        clear_btn = tk.Button(btn_frame, text="Clear", command=lambda: canvas.delete("all"))
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(btn_frame, text="Save & Compare",
                            command=lambda: self.save_drawing(canvas, drawing_window))
        save_btn.pack(side=tk.LEFT, padx=5)
    
    def save_drawing(self, canvas, window):
        """Save drawing"""
        canvas.postscript(file="drawing.eps")
        img = Image.open("drawing.eps")
        img = img.convert('L')
        img.save("drawing.png")
        
        os.remove("drawing.eps")
        window.destroy()
        
        self.display_image("drawing.png")
        self.compare_both_models()
    
    def create_sample_digit(self, digit):
        """Create sample digit"""
        img = np.zeros((280, 280), dtype=np.uint8)
        img.fill(255)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(digit), (80, 180), font, 5, 0, 10)
        
        noise = np.random.randint(-30, 30, (280, 280))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(f"sample_{digit}.png", img)
        self.display_image(f"sample_{digit}.png")
        self.root.after(500, self.compare_both_models)
    
    def show_samples(self):
        """Show sample images window"""
        samples_window = tk.Toplevel(self.root)
        samples_window.title("Sample Digits")
        samples_window.geometry("600x400")
        
        label = tk.Label(samples_window,
                        text="Click a digit button to test it with both models",
                        font=('Arial', 12))
        label.pack(pady=10)
        
        btn_frame = tk.Frame(samples_window)
        btn_frame.pack()
        
        for i in range(10):
            btn = tk.Button(btn_frame,
                          text=str(i),
                          command=lambda d=i: self.create_sample_digit(d),
                          font=('Arial', 14, 'bold'),
                          bg='#3498db',
                          fg='white',
                          width=4,
                          height=2)
            btn.grid(row=i//5, column=i%5, padx=5, pady=5)
    
    def clear_image(self):
        """Clear current image"""
        self.image_label.config(image="", 
                               text="No Image Selected\n\nUse buttons below to input a digit")
        self.current_image = None
        self.recognize_btn.config(state=tk.DISABLED, bg='#e74c3c')
        self.clear_all_results()
    
    def preprocess_image(self, image):
        """Preprocess image for model"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        image = (image - 0.1307) / 0.3081
        
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        return tensor
    
    def predict_with_model(self, image_tensor, model, model_name):
        """Get prediction from a model"""
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            all_probs = probabilities.squeeze().numpy()
            
            return {
                'model': model_name,
                'prediction': prediction.item(),
                'confidence': confidence.item() * 100,
                'probabilities': all_probs,
                'all_predictions': output.squeeze().numpy()
            }
    
    def compare_both_models(self):
        """Compare predictions from both models"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please select or draw an image first!")
            return
        
        # Clear previous results
        self.clear_all_results()
        
        # Preprocess image
        processed = self.preprocess_image(self.current_image)
        
        # Get predictions from both models
        mnist_pred = self.predict_with_model(processed, self.mnist_model, "MNIST Model")
        custom_pred = self.predict_with_model(processed, self.custom_model, "Custom Model")
        
        # Display results in tabs
        self.display_model_results(self.mnist_tab, mnist_pred, '#3498db')
        self.display_model_results(self.custom_tab, custom_pred, '#2ecc71')
        self.display_comparison(self.comparison_tab, mnist_pred, custom_pred)
        
        # Switch to comparison tab
        self.results_notebook.select(2)
    
    def display_model_results(self, parent, prediction, color):
        """Display results for a single model"""
        
        # Clear the tab
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Create canvas for results
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Model title
        canvas.create_text(250, 30,
                          text=prediction['model'],
                          font=('Arial', 16, 'bold'),
                          fill=color)
        
        # Predicted digit (big)
        canvas.create_text(250, 100,
                          text=str(prediction['prediction']),
                          font=('Arial', 72, 'bold'),
                          fill=color)
        
        # Confidence
        conf_text = f"Confidence: {prediction['confidence']:.2f}%"
        conf_color = '#27ae60' if prediction['confidence'] > 90 else '#f39c12' if prediction['confidence'] > 70 else '#e74c3c'
        canvas.create_text(250, 160,
                          text=conf_text,
                          font=('Arial', 14, 'bold'),
                          fill=conf_color)
        
        # Probability bars
        canvas.create_text(250, 200,
                          text="Probability Distribution",
                          font=('Arial', 12, 'bold'),
                          fill='#2c3e50')
        
        # Draw probability bars
        bar_start_y = 220
        bar_height = 100
        bar_spacing = 40
        
        for i in range(10):
            x = 25 + i * bar_spacing
            
            # Bar background
            canvas.create_rectangle(x, bar_start_y, x + 30, bar_start_y + bar_height,
                                   fill='#ecf0f1', outline='#bdc3c7')
            
            # Filled bar
            prob_height = int(prediction['probabilities'][i] * bar_height)
            bar_color = color if i == prediction['prediction'] else '#7f8c8d'
            canvas.create_rectangle(x, bar_start_y + bar_height - prob_height,
                                   x + 30, bar_start_y + bar_height,
                                   fill=bar_color, outline='')
            
            # Digit label
            canvas.create_text(x + 15, bar_start_y + bar_height + 15,
                              text=str(i),
                              font=('Arial', 10, 'bold'),
                              fill='#2c3e50')
            
            # Probability percentage
            prob_percent = prediction['probabilities'][i] * 100
            if prob_percent > 1:
                canvas.create_text(x + 15, bar_start_y + bar_height - prob_height - 10,
                                  text=f"{prob_percent:.1f}%",
                                  font=('Arial', 8),
                                  fill='#2c3e50')
    
    def display_comparison(self, parent, mnist_pred, custom_pred):
        """Display comparison between models"""
        
        # Clear the tab
        for widget in parent.winfo_children():
            widget.destroy()
        
        # Create comparison frame
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Title
        canvas.create_text(250, 30,
                          text="MODEL COMPARISON",
                          font=('Arial', 16, 'bold'),
                          fill='#9b59b6')
        
        # Do models agree?
        agree = mnist_pred['prediction'] == custom_pred['prediction']
        agree_text = "✅ MODELS AGREE" if agree else "⚠️ MODELS DISAGREE"
        agree_color = '#27ae60' if agree else '#e74c3c'
        
        canvas.create_text(250, 70,
                          text=agree_text,
                          font=('Arial', 14, 'bold'),
                          fill=agree_color)
        
        # Predictions side by side
        canvas.create_text(125, 120,
                          text=f"MNIST: {mnist_pred['prediction']}",
                          font=('Arial', 24, 'bold'),
                          fill='#3498db')
        
        canvas.create_text(375, 120,
                          text=f"Custom: {custom_pred['prediction']}",
                          font=('Arial', 24, 'bold'),
                          fill='#2ecc71')
        
        # Confidence comparison
        canvas.create_text(125, 160,
                          text=f"{mnist_pred['confidence']:.1f}%",
                          font=('Arial', 14),
                          fill='#3498db')
        
        canvas.create_text(375, 160,
                          text=f"{custom_pred['confidence']:.1f}%",
                          font=('Arial', 14),
                          fill='#2ecc71')
        
        # Confidence difference
        conf_diff = mnist_pred['confidence'] - custom_pred['confidence']
        diff_text = f"Difference: {abs(conf_diff):.1f}%"
        diff_color = '#27ae60' if abs(conf_diff) < 10 else '#f39c12' if abs(conf_diff) < 20 else '#e74c3c'
        
        canvas.create_text(250, 190,
                          text=diff_text,
                          font=('Arial', 12, 'bold'),
                          fill=diff_color)
        
        # Side-by-side probability comparison
        canvas.create_text(250, 230,
                          text="Probability Comparison",
                          font=('Arial', 12, 'bold'),
                          fill='#2c3e50')
        
        # Draw comparison bars
        bar_start_y = 250
        bar_height = 80
        
        for i in range(10):
            x = 25 + i * 45
            
            # MNIST probability bar
            mnist_prob = mnist_pred['probabilities'][i] * bar_height
            mnist_color = '#3498db' if i == mnist_pred['prediction'] else '#a6c8e3'
            canvas.create_rectangle(x, bar_start_y + bar_height - mnist_prob,
                                   x + 15, bar_start_y + bar_height,
                                   fill=mnist_color, outline='')
            
            # Custom probability bar
            custom_prob = custom_pred['probabilities'][i] * bar_height
            custom_color = '#2ecc71' if i == custom_pred['prediction'] else '#a3e4b3'
            canvas.create_rectangle(x + 17, bar_start_y + bar_height - custom_prob,
                                   x + 32, bar_start_y + bar_height,
                                   fill=custom_color, outline='')
            
            # Digit label
            canvas.create_text(x + 23, bar_start_y + bar_height + 15,
                              text=str(i),
                              font=('Arial', 9, 'bold'),
                              fill='#2c3e50')
        
        # Legend
        legend_y = bar_start_y + bar_height + 50
        
        # MNIST legend
        canvas.create_rectangle(50, legend_y, 70, legend_y + 15, fill='#3498db', outline='')
        canvas.create_text(130, legend_y + 7,
                          text="MNIST Model Prediction",
                          font=('Arial', 9),
                          fill='#3498db',
                          anchor='w')
        
        # Custom legend
        canvas.create_rectangle(50, legend_y + 25, 70, legend_y + 40, fill='#2ecc71', outline='')
        canvas.create_text(130, legend_y + 32,
                          text="Custom Model Prediction",
                          font=('Arial', 9),
                          fill='#2ecc71',
                          anchor='w')
        
        # Analysis
        analysis_y = legend_y + 70
        
        if agree:
            if mnist_pred['confidence'] > 90 and custom_pred['confidence'] > 90:
                analysis = "✅ Both models are very confident and agree!"
            elif mnist_pred['confidence'] > 70 and custom_pred['confidence'] > 70:
                analysis = "⚠️ Models agree but confidence could be higher"
            else:
                analysis = "⚠️ Models agree but confidence is low"
        else:
            if mnist_pred['confidence'] > custom_pred['confidence']:
                analysis = f"⚠️ Models disagree. MNIST is more confident ({mnist_pred['confidence']:.1f}% vs {custom_pred['confidence']:.1f}%)"
            else:
                analysis = f"⚠️ Models disagree. Custom model is more confident ({custom_pred['confidence']:.1f}% vs {mnist_pred['confidence']:.1f}%)"
        
        canvas.create_text(250, analysis_y,
                          text=analysis,
                          font=('Arial', 10, 'bold'),
                          fill=agree_color,
                          width=400,
                          justify=tk.CENTER)
        
        # Export button
        export_btn = tk.Button(canvas,
                              text="💾 Export Comparison",
                              command=lambda: self.export_comparison(mnist_pred, custom_pred),
                              font=('Arial', 10),
                              bg='#9b59b6',
                              fg='white')
        canvas.create_window(250, analysis_y + 40, window=export_btn)
    
    def export_comparison(self, mnist_pred, custom_pred):
        """Export comparison results"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("DUAL MODEL DIGIT RECOGNITION COMPARISON\n")
                f.write("="*60 + "\n\n")
                
                f.write("MODEL INFORMATION:\n")
                f.write("-"*40 + "\n")
                f.write(f"MNIST Model Accuracy: {self.mnist_acc}%\n")
                f.write(f"Custom Model Accuracy: {self.custom_acc}%\n\n")
                
                f.write("RECOGNITION RESULTS:\n")
                f.write("-"*40 + "\n\n")
                
                f.write("MNIST Model:\n")
                f.write(f"  Predicted Digit: {mnist_pred['prediction']}\n")
                f.write(f"  Confidence: {mnist_pred['confidence']:.2f}%\n")
                f.write("  Probabilities:\n")
                for i in range(10):
                    f.write(f"    Digit {i}: {mnist_pred['probabilities'][i]*100:6.2f}%")
                    if i == mnist_pred['prediction']:
                        f.write(" ← PREDICTED")
                    f.write("\n")
                
                f.write("\nCustom Model:\n")
                f.write(f"  Predicted Digit: {custom_pred['prediction']}\n")
                f.write(f"  Confidence: {custom_pred['confidence']:.2f}%\n")
                f.write("  Probabilities:\n")
                for i in range(10):
                    f.write(f"    Digit {i}: {custom_pred['probabilities'][i]*100:6.2f}%")
                    if i == custom_pred['prediction']:
                        f.write(" ← PREDICTED")
                    f.write("\n")
                
                f.write("\nCOMPARISON:\n")
                f.write("-"*40 + "\n")
                agree = mnist_pred['prediction'] == custom_pred['prediction']
                f.write(f"Models Agree: {'Yes' if agree else 'No'}\n")
                if not agree:
                    f.write(f"Disagreement: MNIST={mnist_pred['prediction']}, Custom={custom_pred['prediction']}\n")
                f.write(f"Confidence Difference: {abs(mnist_pred['confidence'] - custom_pred['confidence']):.2f}%\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("End of Report\n")
                f.write("="*60 + "\n")
            
            messagebox.showinfo("Export Successful", f"Comparison saved to:\n{file_path}")

def main():
    """Main function"""
    root = tk.Tk()
    app = DualModelDigitRecognizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()