import streamlit as st
from PIL import Image
import os
import sys
import cv2
import tempfile

# Adjust system path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_preprocessing.gemini_interface import ask_gemini_edit_instructions, match_tones
from image_preprocessing.metadata_extractor import metadata

# ------------------ Streamlit UI Config ------------------ #
st.set_page_config(page_title="AI Image Critique & Style Matching", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-title {
            font-size: 1.25rem;
            color: #34495e;
            margin-top: 1rem;
        }
        .css-1v0mbdj.e115fcil2 {
            background-color: #f9f9f9;
            border-radius: 12px;
            padding: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎨 AI Photo Critique & Style Matching Assistant</div>', unsafe_allow_html=True)

# ------------------ Helper to Save Uploaded Files ------------------ #
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())  # safe and works even after being opened
        return tmp_file.name


# ------------------ Mode Selection ------------------ #
mode = st.radio("Choose a mode:", ("Image Critique", "Match with Sample Image"))

# ------------------ Mode: Image Critique ------------------ #
if mode == "Image Critique":
    st.markdown('<div class="sub-title">Upload an image and describe how you want it improved:</div>', unsafe_allow_html=True)
    prompt = st.text_input("Enter your prompt (e.g., 'Make it look cinematic'):")
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file and prompt:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing and generating suggestions..."):
            temp_path = save_uploaded_file(image_file)

            try:
                img_cv = cv2.imread(temp_path)
                raw_instructions = ask_gemini_edit_instructions(prompt, metadata(img_cv))
            except Exception as e:
                st.error(f"Error processing image: {e}")
                raw_instructions = None
            finally:
                os.remove(temp_path)

        if raw_instructions:
            st.markdown("#### Suggested Edits:")
            st.code(raw_instructions)

# ------------------ Mode: Match with Sample Image ------------------ #
elif mode == "Match with Sample Image":
    st.markdown('<div class="sub-title">Upload a user image and a sample to match styles:</div>', unsafe_allow_html=True)
    prompt = st.text_input("Enter your prompt (e.g., 'Match tones with this reference'):")
    user_img = st.file_uploader("Upload Your Image", type=["png", "jpg", "jpeg"], key="user")
    sample_img = st.file_uploader("Upload Sample Image", type=["png", "jpg", "jpeg"], key="sample")

    if user_img and sample_img and prompt:
        col1, col2 = st.columns(2)

        with col1:
            st.image(user_img, caption="Your Image", use_container_width=True)
        with col2:
            st.image(sample_img, caption="Sample Image", use_container_width=True)

        with st.spinner("Analyzing differences and generating match instructions..."):
            user_temp_path = save_uploaded_file(user_img)
            sample_temp_path = save_uploaded_file(sample_img)

            user_img_cv = cv2.imread(user_temp_path)
            sample_img_cv = cv2.imread(sample_temp_path)

            try:
                instructions = match_tones(
                prompt,
                metadata(user_img_cv),
                metadata(sample_img_cv)
            )
            except Exception as e:
                st.error(f"Error comparing images: {e}")
                instructions = None
            finally:
                os.remove(user_temp_path)
                os.remove(sample_temp_path)

        if instructions:
            st.markdown("#### Style Matching Guidance:")
            st.code(instructions)

# streamlit run ai_photo_enhacer_cv_nd_nlp/app.py