# digit_recognizer_gui.py
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

print("Starting Handwritten Digit Recognizer GUI...")

# Define the CNN model (same as your project)
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

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("900x600")
        
        # Set background color
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.image_path = None
        self.current_image = None
        self.drawing_image = None
        
        # Try to load models
        self.model_mnist = None
        self.model_custom = None
        self.load_models()
        
        # Create GUI
        self.create_widgets()
        
    def load_models(self):
        """Try to load trained models"""
        print("Loading models...")
        
        # Create model instances
        self.model_mnist = SimpleCNN()
        self.model_custom = SimpleCNN()
        
        # Try to load MNIST model
        mnist_paths = ['mnist_model.pth', 'model.pth', 'mnist_cnn.pth']
        mnist_loaded = False
        
        for path in mnist_paths:
            if os.path.exists(path):
                try:
                    self.model_mnist.load_state_dict(torch.load(path, map_location='cpu'))
                    self.model_mnist.eval()
                    mnist_loaded = True
                    print(f"✅ MNIST model loaded from {path}")
                    break
                except:
                    pass
        
        if not mnist_loaded:
            print("⚠️ MNIST model not found, using untrained model")
        
        # Try to load custom model
        custom_paths = ['custom_model.pth', 'model_custom.pth']
        custom_loaded = False
        
        for path in custom_paths:
            if os.path.exists(path):
                try:
                    self.model_custom.load_state_dict(torch.load(path, map_location='cpu'))
                    self.model_custom.eval()
                    custom_loaded = True
                    print(f"✅ Custom model loaded from {path}")
                    break
                except:
                    pass
        
        if not custom_loaded:
            print("⚠️ Custom model not found, using untrained model")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, 
                              text="✏️ Handwritten Digit Recognizer",
                              font=('Arial', 20, 'bold'),
                              bg='#2c3e50',
                              fg='white')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(title_frame,
                                 text="Neural Network Project - Test MNIST and Custom Models",
                                 font=('Arial', 10),
                                 bg='#2c3e50',
                                 fg='#bdc3c7')
        subtitle_label.pack()
        
        # Main Content Frame
        main_frame = tk.Frame(self.root, bg='#ecf0f1')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left Panel - Image Input
        left_panel = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right Panel - Results
        right_panel = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Setup left panel
        self.setup_left_panel(left_panel)
        
        # Setup right panel
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """Setup the left panel with image input controls"""
        
        # Image display label
        self.image_label = tk.Label(parent, 
                                   text="No Image Selected\n\nClick 'Upload' or 'Draw' to begin",
                                   font=('Arial', 12),
                                   bg='white',
                                   fg='#7f8c8d',
                                   relief=tk.SUNKEN,
                                   borderwidth=2)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control buttons frame
        btn_frame = tk.Frame(parent, bg='white')
        btn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Upload button
        upload_btn = tk.Button(btn_frame,
                              text="📁 Upload Image",
                              command=self.upload_image,
                              font=('Arial', 10, 'bold'),
                              bg='#3498db',
                              fg='white',
                              width=15,
                              height=2)
        upload_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Draw button
        draw_btn = tk.Button(btn_frame,
                            text="✏️ Draw Digit",
                            command=self.open_drawing_window,
                            font=('Arial', 10, 'bold'),
                            bg='#2ecc71',
                            fg='white',
                            width=15,
                            height=2)
        draw_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Clear button
        clear_btn = tk.Button(btn_frame,
                             text="🗑️ Clear",
                             command=self.clear_image,
                             font=('Arial', 10, 'bold'),
                             bg='#e74c3c',
                             fg='white',
                             width=15,
                             height=2)
        clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Model selection frame
        model_frame = tk.LabelFrame(parent, 
                                   text="Select Model",
                                   font=('Arial', 11, 'bold'),
                                   bg='white',
                                   padx=10,
                                   pady=10)
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Model selection variable
        self.model_var = tk.StringVar(value="mnist")
        
        # Radio buttons
        mnist_radio = tk.Radiobutton(model_frame,
                                    text="MNIST Model (99.23% accuracy)",
                                    variable=self.model_var,
                                    value="mnist",
                                    font=('Arial', 9),
                                    bg='white')
        mnist_radio.pack(anchor=tk.W, pady=2)
        
        custom_radio = tk.Radiobutton(model_frame,
                                     text="Custom Model (73.33% accuracy)",
                                     variable=self.model_var,
                                     value="custom",
                                     font=('Arial', 9),
                                     bg='white')
        custom_radio.pack(anchor=tk.W, pady=2)
        
        both_radio = tk.Radiobutton(model_frame,
                                   text="Compare Both Models",
                                   variable=self.model_var,
                                   value="both",
                                   font=('Arial', 9),
                                   bg='white')
        both_radio.pack(anchor=tk.W, pady=2)
        
        # Recognize button
        recognize_frame = tk.Frame(parent, bg='white')
        recognize_frame.pack(fill=tk.X, padx=20, pady=20)
        
        recognize_btn = tk.Button(recognize_frame,
                                 text="🔍 Recognize Digit",
                                 command=self.recognize_digit,
                                 font=('Arial', 12, 'bold'),
                                 bg='#9b59b6',
                                 fg='white',
                                 height=2,
                                 width=20)
        recognize_btn.pack()
        
        # Sample digits
        sample_frame = tk.LabelFrame(parent,
                                    text="Try Sample Digits",
                                    font=('Arial', 11, 'bold'),
                                    bg='white',
                                    padx=10,
                                    pady=10)
        sample_frame.pack(fill=tk.X, padx=20, pady=10)
        
        sample_btn_frame = tk.Frame(sample_frame, bg='white')
        sample_btn_frame.pack()
        
        # Create buttons for digits 0-9
        for i in range(10):
            btn = tk.Button(sample_btn_frame,
                          text=str(i),
                          command=lambda d=i: self.create_sample_digit(d),
                          font=('Arial', 10, 'bold'),
                          bg='#f39c12',
                          fg='white',
                          width=3)
            btn.grid(row=i//5, column=i%5, padx=2, pady=2)
    
    def setup_right_panel(self, parent):
        """Setup the right panel for displaying results"""
        
        # Results label
        results_label = tk.Label(parent,
                                text="Recognition Results",
                                font=('Arial', 14, 'bold'),
                                bg='white')
        results_label.pack(pady=10)
        
        # Results display frame
        self.results_frame = tk.Frame(parent, bg='white')
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize with empty message
        self.clear_results()
    
    def clear_results(self):
        """Clear the results display"""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        empty_label = tk.Label(self.results_frame,
                              text="Results will appear here\nafter recognizing a digit",
                              font=('Arial', 12),
                              bg='white',
                              fg='#95a5a6')
        empty_label.pack(expand=True)
    
    def upload_image(self):
        """Upload an image file"""
        file_path = filedialog.askopenfilename(
            title="Select a digit image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if file_path:
            self.display_image(file_path)
    
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Resize to fit in the display area
            img.thumbnail((300, 300))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Store for processing
            self.current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.image_path = image_path
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def open_drawing_window(self):
        """Open a window for drawing digits"""
        drawing_window = tk.Toplevel(self.root)
        drawing_window.title("Draw a Digit")
        drawing_window.geometry("400x500")
        
        # Drawing canvas
        self.drawing_canvas = tk.Canvas(drawing_window, 
                                       width=280, 
                                       height=280, 
                                       bg='white',
                                       cursor='cross')
        self.drawing_canvas.pack(pady=10)
        
        # Drawing variables
        self.last_x = None
        self.last_y = None
        self.drawing_image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.drawing_image)
        
        # Bind mouse events
        self.drawing_canvas.bind("<Button-1>", self.start_drawing)
        self.drawing_canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.drawing_canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Control buttons
        control_frame = tk.Frame(drawing_window)
        control_frame.pack(pady=10)
        
        clear_btn = tk.Button(control_frame,
                             text="Clear",
                             command=self.clear_drawing,
                             width=10)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(control_frame,
                            text="Save & Recognize",
                            command=lambda: self.save_drawing(drawing_window),
                            width=15,
                            bg='#2ecc71',
                            fg='white')
        save_btn.pack(side=tk.LEFT, padx=5)
    
    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_on_canvas(self, event):
        if self.last_x and self.last_y:
            # Draw on canvas
            self.drawing_canvas.create_line(self.last_x, self.last_y,
                                           event.x, event.y,
                                           width=15,
                                           fill='black',
                                           capstyle=tk.ROUND,
                                           smooth=True)
            
            # Draw on image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill='black',
                          width=15)
            
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_drawing(self):
        self.drawing_canvas.delete("all")
        self.drawing_image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.drawing_image)
    
    def save_drawing(self, window):
        """Save the drawing and close window"""
        if self.drawing_image:
            # Convert to grayscale
            gray_image = self.drawing_image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(gray_image)
            
            # Store for processing
            self.current_image = img_array
            
            # Display in main window
            img_resized = gray_image.copy()
            img_resized.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img_resized)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            # Close drawing window
            window.destroy()
            
            # Auto-recognize
            self.root.after(500, self.recognize_digit)
    
    def create_sample_digit(self, digit):
        """Create a sample digit image"""
        # Create white background
        img = np.ones((280, 280), dtype=np.uint8) * 255
        
        # Draw digit
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(digit), (80, 180), font, 5, 0, 10)
        
        # Add some noise to look handwritten
        noise = np.random.randint(-30, 30, (280, 280))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Display
        pil_img = Image.fromarray(img)
        pil_img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(pil_img)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
        
        # Store for processing
        self.current_image = img
        
        # Auto-recognize
        self.root.after(500, self.recognize_digit)
    
    def clear_image(self):
        """Clear the current image"""
        self.image_label.config(image="", 
                               text="No Image Selected\n\nClick 'Upload' or 'Draw' to begin")
        self.current_image = None
        self.clear_results()
    
    def preprocess_image(self, image):
        """Preprocess image for the model"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))
        
        # Normalize like MNIST
        image = image.astype('float32') / 255.0
        image = (image - 0.1307) / 0.3081
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        return image_tensor
    
    def recognize_digit(self):
        """Recognize the digit in the current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please select or draw an image first!")
            return
        
        # Clear previous results
        self.clear_results()
        
        # Preprocess image
        processed = self.preprocess_image(self.current_image)
        
        # Get model selection
        model_choice = self.model_var.get()
        
        # Get predictions
        predictions = []
        
        if model_choice == "mnist" or model_choice == "both":
            if self.model_mnist:
                pred = self.predict_with_model(processed, self.model_mnist, "MNIST Model")
                predictions.append(pred)
        
        if model_choice == "custom" or model_choice == "both":
            if self.model_custom:
                pred = self.predict_with_model(processed, self.model_custom, "Custom Model")
                predictions.append(pred)
        
        # Display results
        self.display_recognition_results(predictions)
    
    def predict_with_model(self, image_tensor, model, model_name):
        """Get prediction from a specific model"""
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            # Get all probabilities
            all_probs = probabilities.squeeze().numpy()
            
            return {
                'model_name': model_name,
                'prediction': prediction.item(),
                'confidence': confidence.item() * 100,
                'probabilities': all_probs
            }
    
    def display_recognition_results(self, predictions):
        """Display the recognition results"""
        
        if not predictions:
            error_label = tk.Label(self.results_frame,
                                  text="No predictions available\nCheck if models are loaded",
                                  font=('Arial', 12),
                                  fg='red',
                                  bg='white')
            error_label.pack(expand=True)
            return
        
        # Create a notebook for multiple predictions
        if len(predictions) > 1:
            notebook = ttk.Notebook(self.results_frame)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            for i, pred in enumerate(predictions):
                frame = tk.Frame(notebook, bg='white')
                notebook.add(frame, text=pred['model_name'])
                self.create_result_display(frame, pred)
        else:
            # Single prediction
            self.create_result_display(self.results_frame, predictions[0])
    
    def create_result_display(self, parent, prediction):
        """Create display for a single prediction"""
        
        # Main result frame
        result_frame = tk.Frame(parent, bg='white')
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model name
        model_label = tk.Label(result_frame,
                              text=prediction['model_name'],
                              font=('Arial', 14, 'bold'),
                              bg='white',
                              fg='#2c3e50')
        model_label.pack(pady=5)
        
        # Prediction result
        pred_frame = tk.Frame(result_frame, bg='#ecf0f1', relief=tk.RAISED, borderwidth=2)
        pred_frame.pack(fill=tk.X, pady=10, padx=20)
        
        pred_label = tk.Label(pred_frame,
                             text=f"Predicted Digit:",
                             font=('Arial', 12),
                             bg='#ecf0f1')
        pred_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Big digit display
        digit_label = tk.Label(pred_frame,
                              text=str(prediction['prediction']),
                              font=('Arial', 36, 'bold'),
                              bg='#ecf0f1',
                              fg='#e74c3c')
        digit_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Confidence
        conf_value = prediction['confidence']
        conf_color = '#27ae60' if conf_value > 90 else '#f39c12' if conf_value > 70 else '#e74c3c'
        
        conf_frame = tk.Frame(result_frame, bg='white')
        conf_frame.pack(fill=tk.X, pady=5, padx=20)
        
        conf_label = tk.Label(conf_frame,
                             text=f"Confidence: {conf_value:.2f}%",
                             font=('Arial', 12),
                             bg='white',
                             fg=conf_color)
        conf_label.pack()
        
        # Confidence bar
        conf_bar_frame = tk.Frame(result_frame, bg='white')
        conf_bar_frame.pack(fill=tk.X, pady=5, padx=20)
        
        # Create confidence bar
        canvas = tk.Canvas(conf_bar_frame, height=20, bg='white', highlightthickness=0)
        canvas.pack(fill=tk.X)
        
        bar_width = 200
        fill_width = int(bar_width * (conf_value / 100))
        
        # Draw bar background
        canvas.create_rectangle(0, 0, bar_width, 20, fill='#ecf0f1', outline='')
        
        # Draw filled portion
        canvas.create_rectangle(0, 0, fill_width, 20, fill=conf_color, outline='')
        
        # Probability distribution
        prob_frame = tk.LabelFrame(result_frame,
                                  text="Probability Distribution",
                                  font=('Arial', 11, 'bold'),
                                  bg='white',
                                  padx=10,
                                  pady=10)
        prob_frame.pack(fill=tk.X, pady=10, padx=20)
        
        # Create grid of probabilities
        for i in range(10):
            prob = prediction['probabilities'][i] * 100
            
            # Frame for each digit
            digit_prob_frame = tk.Frame(prob_frame, bg='white')
            digit_prob_frame.grid(row=i//5, column=i%5, padx=5, pady=5)
            
            # Digit label
            digit_label = tk.Label(digit_prob_frame,
                                  text=f"{i}:",
                                  font=('Arial', 9),
                                  bg='white',
                                  width=3)
            digit_label.pack(side=tk.LEFT)
            
            # Probability label
            prob_color = '#e74c3c' if i == prediction['prediction'] else '#7f8c8d'
            prob_label = tk.Label(digit_prob_frame,
                                 text=f"{prob:.1f}%",
                                 font=('Arial', 9, 'bold' if i == prediction['prediction'] else 'normal'),
                                 bg='white',
                                 fg=prob_color,
                                 width=6)
            prob_label.pack(side=tk.LEFT)
        
        # Interpretation
        interpret_frame = tk.Frame(result_frame, bg='#f8f9fa', relief=tk.SUNKEN, borderwidth=1)
        interpret_frame.pack(fill=tk.X, pady=10, padx=20)
        
        if conf_value > 90:
            interpretation = "✅ Excellent recognition! The model is very confident."
        elif conf_value > 70:
            interpretation = "⚠️ Good recognition, but could be clearer."
        else:
            interpretation = "❓ Low confidence. Try drawing clearer or use a different image."
        
        interpret_label = tk.Label(interpret_frame,
                                  text=interpretation,
                                  font=('Arial', 10),
                                  bg='#f8f9fa',
                                  wraplength=400,
                                  justify=tk.LEFT)
        interpret_label.pack(padx=10, pady=10)

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()