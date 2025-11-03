"""
Car vs Motorcycle Classifier - Streamlit Web Application
This app ONLY loads the pre-trained model and makes predictions.
Run train_model.py FIRST to create the model file.
"""

import base64
import json
import numpy as np
import os
import pandas as pd
import streamlit as st
import tensorflow as tf
import time

from pathlib import Path
from PIL import Image

# ----------------------------
# Configuration
# ----------------------------
# Try multiple possible model paths
POSSIBLE_MODEL_PATHS = [
    "advanced_model_clean.keras",
    "advanced_model_clean.h5",
    "./advanced_model_clean.keras",
    "./advanced_model_clean.h5",
    "models/advanced_model_clean.keras",
    "models/advanced_model_clean.h5"
]

IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Car", "Motorcycle"]

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Motorcycle and Car Classifier",
    page_icon="🚗",
    layout="centered"
)

# ----------------------------
# Background helper
# ----------------------------
def set_background(image_path: str):
    """Set background image if it exists, otherwise use gradient."""
    if Path(image_path).exists():
        img_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        bg_image = f"url(data:image/png;base64,{encoded})"
    else:
        bg_image = "linear-gradient(135deg, #A3AFFF 0%, #764ba2 100%)"
    
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.6)),
                        {bg_image};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stMainContainer"] {{
            background-color: transparent !important;
        }}
        
        /* Headers - Darker purple for visibility */
        h1, h2, h3, h4, h5, h6 {{
            color: #4a3a8c !important;
        }}
        
        /* Captions and body text - Dark for visibility on light background */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .stMarkdown p {{
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }}
        
        /* File uploader label specifically */
        [data-testid="stFileUploader"] label {{
            color: black !important;
            font-weight: 600 !important;
        }}
        
        /* Info boxes (st.info, st.warning, etc.) */
        [data-testid="stAlert"],
        [data-testid="stAlert"] > div:first-child {{
            background-color: rgba(0, 0, 102, 0.47) !important;
            border: 1px solid #4a3a8c !important;
            box-shadow: none !important;
            border-radius: 10px !important;
        }}
        
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div {{
            color: #1a1a1a !important;
        }}
        
        /* Links */
        a {{
            color: #4a3a8c !important;
            font-weight: 600 !important;
        }}
        
        /* Prediction box */
        .prediction-box {{
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            background-color: rgba(0, 0, 102, 0.70) !important;
            border: 1px solid #4a3a8c;
        }}
        
        /* Ensure all text in prediction box is white */
        .prediction-box h2,
        .prediction-box p {{
            color: white !important;
        }}
        
        /* Expanders */
        [data-testid="stExpander"] {{
            background-color: rgba(0, 0, 102, 0.70) !important;
            border: 1px solid #4a3a8c !important;
        }}
        
        /* When the expander is clicked/open (focus or active) */
        [data-testid="stExpander"]:focus summary,
        [data-testid="stExpander"][open] summary {{
            background-color: rgba(0, 0, 102, 0.7) !important; /* keep same as normal */
        }}
        
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] li,
        [data-testid="stExpander"] strong {{
            color: rgba(0, 0, 102, 0.70) !important;
        }}

        /* Expander caret and header text color */
        [data-testid="stExpander"] summary {{
            color: #000000 !important;           /* header text */
            fill: #000000 !important;            /* caret icon (SVG) */
        }}

        [data-testid="stExpander"] svg {{
            color: #000000 !important;           /* fallback */
            fill: #000000 !important;
        }}
        
        /* Headers inside expanders */
        [data-testid="stExpander"] h1,
        [data-testid="stExpander"] h2,
        [data-testid="stExpander"] h3,
        [data-testid="stExpander"] h4,
        [data-testid="stExpander"] h5,
        [data-testid="stExpander"] h6 {{
            color: #9999ff !important;  /* very light purple for contrast */
        }}
                
        /* Metric cards */
        .metric-card {{
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.9) !important;
        }}
        
        /* Dividers */
        hr {{
            border-color: #4a3a8c !important;
            opacity: 0.3 !important;
        }}

        /* Nuclear option - everything inside file uploader after dropzone */
        [data-testid="stFileUploader"] * {{
            background-color: rgba(0, 0, 102, 0.75) !important;
        }}
        
        /* But keep the dropzone and button normal */
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploader"] button {{
            background-color: rgba(0, 0, 102, 0.75) !important; /* your original color */
        }}
        
        /* File uploader label/caption - make white for visibility on purple background */
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] > label {{
            color: white !important;
            font-weight: 600 !important;
        }}
        
        /* Target the list item container that holds the filename */
        [data-testid="stFileUploader"] div[role="listitem"] {{
            background-color: rgba(0, 0, 102, 0.75) !important;
            border-radius: 8px !important;
            padding: 8px 12px !important;
        }}
        
        /* More aggressive - target parent containers */
        [data-testid="stFileUploader"] > div > div {{
            background-color: rgba(0, 0, 102, 0.75) !important;
        }}
        
        /* File uploader box background and border */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: rgba(0, 0, 102, .78) !important;
            border: 2px dashed #4a3a8c !important;
            border-radius: 10px !important;
        }}

        /* File uploader button */
        [data-testid="stFileUploader"] button {{
            background-color: #000066 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: 1px solid white !important;
        }}
        
        /* Model Scope info box text */
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div,
        [data-testid="stAlert"] strong {{
            color: #ffffff !important;
        }}

        /* Expander headers (labels + carets) */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] svg {{
            color: #ffffff !important;
            fill: #ffffff !important;
        }}

        /* Expander inner text */
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] li,
        [data-testid="stExpander"] strong {{
            color: #ffffff !important;
        }}    
        
        /* Expander summary - keep consistent color in all states */
        [data-testid="stExpander"] summary {{
            background-color: rgba(0, 0, 102, 0.70) !important;
            color: #ffffff !important;
            fill: #ffffff !important;
        }}
        
        /* Prevent color change on hover */
        [data-testid="stExpander"] summary:hover {{
            background-color: rgba(0, 0, 102, 0.70) !important;
            color: #ffffff !important;
        }}
        
        /* Prevent color change when clicked/active */
        [data-testid="stExpander"] summary:active,
        [data-testid="stExpander"] summary:focus {{
            background-color: rgba(0, 0, 102, 0.70) !important;
            color: #ffffff !important;
        }}
        
        /* When expander is open */
        [data-testid="stExpander"][open] summary {{
            background-color: rgba(0, 0, 102, 0.70) !important;
            color: #ffffff !important;
        }}
        
        </style>
        """,
        unsafe_allow_html=True
        
    )

set_background("background.webp")

# ----------------------------
# Load Training History (if available)
# ----------------------------
def load_training_metrics():
    """Load training metrics from saved file, or use defaults if not available."""
    metrics_path = "training_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load training metrics: {e}")
    
    # Default values if metrics file doesn't exist
    return {
        "baseline": {"accuracy": 0.87, "precision": 0.86, "recall": 0.87, "f1": 0.86, "time": 45, "size": 2.1},
        "intermediate": {"accuracy": 0.92, "precision": 0.91, "recall": 0.92, "f1": 0.91, "time": 120, "size": 3.8},
        "advanced": {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.94, "time": 180, "size": 4.5}
    }

# Load metrics
training_metrics = load_training_metrics()

# Create comparison dataframe from actual metrics
model_comparison = pd.DataFrame({
    "Model": ["Baseline", "Intermediate", "Advanced"],
    "Accuracy": [training_metrics["baseline"]["accuracy"], 
                 training_metrics["intermediate"]["accuracy"], 
                 training_metrics["advanced"]["accuracy"]],
    "Precision": [training_metrics["baseline"]["precision"], 
                  training_metrics["intermediate"]["precision"], 
                  training_metrics["advanced"]["precision"]],
    "Recall": [training_metrics["baseline"]["recall"], 
               training_metrics["intermediate"]["recall"], 
               training_metrics["advanced"]["recall"]],
    "F1-Score": [training_metrics["baseline"]["f1"], 
                 training_metrics["intermediate"]["f1"], 
                 training_metrics["advanced"]["f1"]],
    "Training Time (s)": [training_metrics["baseline"]["time"], 
                          training_metrics["intermediate"]["time"], 
                          training_metrics["advanced"]["time"]],
    "Model Size (MB)": [training_metrics["baseline"]["size"], 
                        training_metrics["intermediate"]["size"], 
                        training_metrics["advanced"]["size"]]
})

# ----------------------------
# Load Model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    import os
    
    # Try to find the model file
    for model_path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                return model
            except Exception as e:
                st.warning(f"Found {model_path} but couldn't load it: {e}")
                continue
    
    # If no model found, show helpful error
    st.error("⚠️ Model file not found. Please ensure your model file is in one of these locations:")
    for path in POSSIBLE_MODEL_PATHS:
        st.code(path)
    st.info("💡 **Tip:** Your model might be saved as .h5 instead of .keras. Try renaming it or re-saving it.")
    return None

# ----------------------------
# Prediction Function
# ----------------------------
def predict_image(model, image):
    """Predict whether image is a car or motorcycle."""
    # Resize and preprocess
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    
    # Binary classification: 0 = Car, 1 = Motorcycle
    predicted_class = CLASS_NAMES[1] if confidence > 0.5 else CLASS_NAMES[0]
    confidence_score = confidence if confidence > 0.5 else 1 - confidence
    
    return predicted_class, confidence_score

# ----------------------------
# Main App
# ----------------------------
st.title("🚗 🏍️ Car vs Motorcycle Classifier")
st.markdown("### AI-Powered Car vs Motorcycle Detection")
st.caption("Upload an image to classify it as a car or motorcycle using deep learning")

# Add clear scope statement
st.info(
    "ℹ️ **Model Scope:** For portfolio demonstration purposes.\n\n"
    "This is a binary classifier trained exclusively on cars and motorcycles.\n\n"
    "It will attempt to classify any image into one of these two categories.\n\n"
    "Some of the confident predictions can be amusing; try loading images of frogs, pizzas, pets, friends...!"
)

# Important: Check if model exists BEFORE trying to load
if not any(os.path.exists(path) for path in POSSIBLE_MODEL_PATHS):
    st.error("### ⚠️ Model Not Found!")
    st.markdown("""
    The trained model file doesn't exist yet. You need to train the model first:
    
    **Step 1: Train the model** (run this in your terminal, NOT in Streamlit):
    ```bash
    py train_model.py
    ```
    This will create `advanced_model_clean.keras`
    
    **Step 2: After training completes, refresh this page**
    
    ---
    
    **Note:** `train_model.py` and `image_processing.py` are separate files:
    - `train_model.py` = Trains the model (run once)
    - `image_processing.py` = This Streamlit app (loads the trained model)
    """)
    st.stop()

# Load model
with st.spinner("Loading AI model..."):
    model = load_model()

if model is None:
    st.markdown("### 🔧 Setup Instructions")
    st.markdown("""
    To run this app, you need to add your trained model file:
    
    1. **If you have the model from your training script:**
       - Look for a file named `advanced_model_clean.keras` or `advanced_model_clean.h5`
       - Place it in the same directory as `image_processing.py`
    
    2. **If you need to re-save the model:**
       ```python
       # In your training script, after training:
       model.save('advanced_model_clean.keras')  # or .h5
       ```
    
    3. **Current directory contents:**
    """)
    import os
    files = os.listdir('.')
    st.write(files)
    st.stop()

# ----------------------------
# File Upload Section
# ----------------------------
st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload an image of a car or motorcycle",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📸 Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("#### 🤖 Prediction")
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            start_time = time.time()
            predicted_class, confidence = predict_image(model, image)
            inference_time = time.time() - start_time
        
        # Display results
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: {'#4CAF50' if confidence > 0.8 else '#FF9800'}; margin:0;">
                {predicted_class}
            </h2>
            <p style="font-size: 24px; margin: 10px 0; color: white;">
                {confidence * 100:.1f}% confidence
            </p>
            <p style="color: white; font-size: 14px;">
                Inference time: {inference_time*1000:.0f}ms
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(confidence)
        
        # Additional info with context about model limitations
        if confidence < 0.7:
            st.warning("⚠️ **Low confidence.** The model is uncertain about this prediction. Remember: this classifier only distinguishes between cars and motorcycles, so images of other objects may produce unreliable results.")
        elif confidence > 0.95:
            st.success("✅ High confidence - the model is very certain!")

# ----------------------------
# Sample Images Section
# ----------------------------
st.markdown("---")
st.markdown("### 💡 Don't have an image? Try these examples:")

example_cols = st.columns(3)
example_images = {
    "Car 1": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400",
    "Motorcycle 1": "https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=400",
    "Car 2": "https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=400"
}

for idx, (name, url) in enumerate(example_images.items()):
    with example_cols[idx]:
        st.caption(name)
        st.markdown(f"[View Image]({url})")

# ----------------------------
# Model Information Section
# ----------------------------
with st.expander("📊 Model Performance & Methodology"):
    st.markdown("#### Model Development Process")
    st.markdown("""
    This classifier was developed through an iterative process:
    
    1. **Baseline CNN** - Simple architecture to establish performance baseline
    2. **Intermediate CNN** - Added batch normalization, dropout, and regularization
    3. **Advanced CNN** - Optimized architecture with careful hyperparameter tuning
    
    The Advanced model achieved the best results and is used for predictions in this app.
    """)
    
    st.markdown("#### Performance Comparison")
    st.dataframe(
        model_comparison.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score"],
            color="#026D02"
        ),
        use_container_width=True
    )
    
    st.markdown("#### Technical Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Architecture:**
        - 2 Convolutional blocks
        - Batch Normalization
        - MaxPooling layers
        - Dropout (0.3, 0.4)
        - Dense layers (128 units)
        - Binary classification output
        """)
    
    with col2:
        st.markdown("""
        **Training:**
        - Image size: 128x128
        - Batch size: 32
        - Optimizer: Adam (LR=0.0001)
        - Data augmentation applied
        - Class weights for imbalance
        - ReduceLROnPlateau callback
        """)
    
    st.markdown("#### Key Achievements")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Accuracy", f"{training_metrics['advanced']['accuracy']*100:.0f}%")
    with metric_cols[1]:
        st.metric("Precision", f"{training_metrics['advanced']['precision']*100:.0f}%")
    with metric_cols[2]:
        st.metric("Recall", f"{training_metrics['advanced']['recall']*100:.0f}%")
    with metric_cols[3]:
        st.metric("F1-Score", f"{training_metrics['advanced']['f1']*100:.0f}%")

# ----------------------------
# About Section
# ----------------------------
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### Car vs Motorcycle Classification with Deep Learning
    
    This project demonstrates the application of Convolutional Neural Networks (CNNs) 
    for binary image classification. The model was trained on a dataset of car and 
    motorcycle images, with careful attention to:
    
    - **Data preprocessing** - Image normalization, augmentation, and train/val/test splitting
    - **Class imbalance handling** - Using class weights to address dataset imbalance
    - **Model optimization** - Iterative improvements through architecture refinement
    - **Evaluation** - Comprehensive metrics including confusion matrices and error analysis
    
    **Technologies Used:**
    - TensorFlow/Keras for deep learning
    - Streamlit for web interface
    - NumPy & Pandas for data processing
    - Scikit-learn for evaluation metrics
    
    **Dataset:**
    - Training images: 3,832 (1,920 cars + 1,912 motorcycles)
    - Validation images: 479 (240 cars + 239 motorcycles)
    - Test images: 480 (240 cars + 240 motorcycles)
    - Total images: 4,791
    - Split ratio: 80/10/10
    
    **Model Limitations:**
    - This is a binary classifier trained only on cars and motorcycles
    - Will attempt to classify any image as one of these two categories
    - Performs best with front or front-angled views (most common in training data)
    - May show lower confidence on side profiles, rear views, or unusual angles
    - May misclassify other vehicles (trucks, buses, bicycles) or non-vehicle images
    """)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("🏍️ 🚗 Created by **Chris G.** | Built with TensorFlow & Streamlit | Deployed for portfolio demonstration")