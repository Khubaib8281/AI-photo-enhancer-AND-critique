import streamlit as st
from PIL import Image
import os
import sys
import cv2
import tempfile

# Local module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_preprocessing.gemini_interface import ask_gemini_edit_instructions, match_tones
from image_preprocessing.metadata_extractor import metadata

# ------------------ Streamlit Config ------------------ #
st.set_page_config(page_title="VisionTune | AI Photo Enhancer", layout="wide")

# ------------------ Custom CSS ------------------ #
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #f0f2f5;
            color: black;
        }

        .main-title {
            font-weight: 900;
            color: #1e293b;
            text-align: center;
            margin-top: 2rem;
            margin-bottom: 1.5rem;
            font-size: 2.5rem;
        }

        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem;
            }
        }

        @media (max-width: 480px) {
            .main-title {
                font-size: 1.6rem;
            }
        }


        .sub-title {
            font-size: 1.2rem;
            color: #475569;
            margin-top: 1.5rem;
        }

        label[data-baseweb="radio"] {
            color: black;
            background-color: black ;
            border-radius: 6px;
            # display: inline-block;
        }

        .stRadio > div {
            justify-content: center;
            background-color: black !important;
            color: black !important;
            padding: 10px;
            border-radius: 10px;
        }    
        
        .stRadio > div {
            justify-content: center;
            background-color: black;
            color : black;
        }

        .stTextInput > div > input {
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            color: black;
        }

        .stFileUploader {
            background-color: black;
            border: 1px dashed #94a3b8;
            border-radius: 10px;
            color : black;
            padding: 1rem;
        }

        .stButton > button {
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 8px;
            transition: 0.3s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #1e40af;
        }
            
        .st.spinner{
            color : black
        }

        .block-container {
            padding-top: 2rem;
        }

        code {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.9rem;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f0f2f6;
            color: #333;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }

    </style>
    <div class = "footer">
        Developed with ❤️ by <strong>Muhammad Khubaib Ahmad</strong> • © 2025 All rights reserved
    </div>

""", unsafe_allow_html=True)

# ------------------ Title ------------------ #
st.markdown('<div class="main-title">🎨 VisionTune: AI Photo Critique & Style Matching</div>', unsafe_allow_html=True)

# ------------------ File Save Utility ------------------ #
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

# ------------------ Mode Selector ------------------ #
mode = st.radio("Select Mode", ["🔍 Image Critique", "🎯 Match with Sample Image"], horizontal=True)

# ------------------ Image Critique Mode ------------------ #
if mode == "🔍 Image Critique":
    st.markdown('<div class="sub-title">Upload an image and describe how you want it improved:</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 4])
    with col1:
        prompt = st.text_input("Enhancement Prompt", placeholder="e.g., Make it look cinematic")
    with col2:
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image_file and prompt:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="📸 Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Analyzing and generating recommendations..."):
            temp_path = save_uploaded_file(image_file)
            try:
                img_cv = cv2.imread(temp_path)
                raw_instructions = ask_gemini_edit_instructions(prompt, metadata(img_cv))
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                raw_instructions = None
            finally:
                os.remove(temp_path)

        if raw_instructions:
            st.divider()
            st.markdown("### ✨ Suggested Edits")
            st.code(raw_instructions, language="markdown")

# ------------------ Match Style Mode ------------------ #
elif mode == "🎯 Match with Sample Image":
    st.markdown('<div class="sub-title">Upload your image and a reference to match tones/style:</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        user_img = st.file_uploader("📷 Upload Your Image", type=["jpg", "jpeg", "png"], key="user")
    with col2:
        sample_img = st.file_uploader("🎯 Upload Sample Image", type=["jpg", "jpeg", "png"], key="sample")

    prompt = st.text_input("Style Matching Prompt", placeholder="e.g., Match tones with this reference")

    if user_img and sample_img and prompt:
        col1, col2 = st.columns(2)
        with col1:
            st.image(user_img, caption="👤 Your Image", use_container_width=True)
        with col2:
            st.image(sample_img, caption="🎨 Sample Image", use_container_width=True)

        with st.spinner("⚙️ Comparing and generating style guidance..."):
            user_temp_path = save_uploaded_file(user_img)
            sample_temp_path = save_uploaded_file(sample_img)

            try:
                user_img_cv = cv2.imread(user_temp_path)
                sample_img_cv = cv2.imread(sample_temp_path)
                instructions = match_tones(prompt, metadata(user_img_cv), metadata(sample_img_cv))
            except Exception as e:
                st.error(f"❌ Error comparing images: {e}")
                instructions = None
            finally:
                os.remove(user_temp_path)
                os.remove(sample_temp_path)

        if instructions:
            st.divider()
            st.markdown("### 🧩 Style Matching Instructions")
            st.code(instructions, language="markdown")


# AesthetiQ: AI-Powered Visual Critique
# streamlit run ai_photo_enhacer_cv_nd_nlp/app.py