import streamlit as st
import os
import sys
from PIL import Image
import pandas as pd

# Add parent directory to sys.path to allow imports from Backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.predict import load_trained_model, predict_tumor
from Backend.config import OUTPUT_DIR

# -----------------
# Page Configuration
# -----------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# -----------------
# Load Custom CSS
# -----------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), 'style.css')
if os.path.exists(css_path):
    load_css(css_path)

# -----------------
# App UI
# -----------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1><span class="icon">🧠</span> Brain Tumor Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an MRI scan and let AI analyze it instantly.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main Card/Container
st.markdown('<div class="upload-card">', unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Drag & drop your MRI image here or browse files", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, use_column_width=True)

    # Predict Button
    submit_button = st.button("Predict Tumor")

    if submit_button:
        with st.spinner("Analyzing MRI scan..."):
            # Load the model
            model_path = os.path.join(OUTPUT_DIR, 'brain_tumor_model.keras')
            try:
                model = load_trained_model(model_path)
                
                # Make prediction
                pred_class, confidences = predict_tumor(model, image)
                
                st.success("Analysis Complete!")
                
                # Display Prediction result
                st.markdown(f"<h2 style='text-align: center; color: white;'>Predicted: <span style='color: #00d2ff;'>{pred_class.upper()}</span></h2>", unsafe_allow_html=True)
                
                # Display Confidences as a Bar Chart
                st.subheader("Confidence Scores")
                conf_df = pd.DataFrame(
                    list(confidences.values()), 
                    index=list(confidences.keys()), 
                    columns=['Confidence']
                )
                conf_df = conf_df * 100 # Convert to percentage
                st.bar_chart(conf_df)
                
            except Exception as e:
                st.error(f"Error loading model or making prediction. Please make sure the model is trained first! Details: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #888;">
    <small>Using CNN for Brain Tumor Classification. For educational purposes only.</small>
</div>
""", unsafe_allow_html=True)
