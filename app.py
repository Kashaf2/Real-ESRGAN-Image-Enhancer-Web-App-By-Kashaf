import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from io import BytesIO

# Configure Streamlit page
st.set_page_config(page_title="Real-ESRGAN Image Enhancer", layout="centered")
st.title("🖼️ Real-ESRGAN Image Enhancer")
st.markdown("Upload a model, select scale, and enhance images — all inside this app!")

# Sidebar: Upload model
st.sidebar.header("1. Upload RealESRGAN Model (.pth)")
model_file = st.sidebar.file_uploader("Upload model weights", type=["pth"])

# Sidebar: Select scale factor
scale = st.sidebar.selectbox("2. Choose Upscale Factor", [2, 4], index=1)

# Function to detect model architecture from .pth file
def detect_model_architecture(model_path):
    try:
        # Load the checkpoint from the provided model path
        ckpt = torch.load(model_path, map_location='cpu')
        
        # Check for RealESRGAN-related keys (RealESRGAN or SRVGGNet)
        if 'params_ema' in ckpt:
            ckpt = ckpt['params_ema']
        elif 'params' in ckpt:
            ckpt = ckpt['params']
        
        # Check for the key pattern that is common to RealESRGAN architectures
        if 'body.0.weight' in ckpt:
            return 'RRDBNet'  # Standard RealESRGAN architecture
        elif 'generator' in ckpt:
            return 'SRVGGNet'  # SRVGGNet architecture (older or experimental RealESRGAN model)
        else:
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error detecting model architecture: {e}")
        return None

# Define the architecture for RRDBNet (RealESRGAN models)
def get_architecture(arch_name):
    if arch_name == 'RRDBNet':
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    elif arch_name == 'SRVGGNet':
        # If you want to load SRVGGNet or any other custom architecture, add it here
        return None  # Returning None as placeholder for unsupported models in this case
    else:
        return None

# Load model if ready
upsampler = None
if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp.write(model_file.read())
        tmp_model_path = tmp.name

    try:
        # Detect the architecture
        model_arch = detect_model_architecture(tmp_model_path)
        if model_arch is None:
            st.sidebar.error("❌ Unknown or unsupported model architecture.")
        else:
            # Load model based on detected architecture
            model = get_architecture(model_arch)

            if model is None:
                st.sidebar.error(f"❌ Could not create model for architecture {model_arch}.")
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                upsampler = RealESRGANer(
                    scale=scale,
                    model_path=tmp_model_path,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=(device == 'cuda'),
                    device=torch.device(device)
                )
                st.sidebar.success(f"✅ {model_arch} model loaded successfully!")

    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
        upsampler = None

# Main: Upload image
st.header("4. Upload an Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Enhance image
if uploaded_image and upsampler:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Original Image", use_column_width=True)

    # Add a progress bar to show enhancement progress
    progress_bar = st.progress(0)
    progress_text = st.empty()

    if st.button("🚀 Enhance Image"):
        try:
            img_np = np.array(img)

            # Simulate the enhancement process with progress updates
            for i in range(1, 101):
                progress_bar.progress(i)
                progress_text.text(f"Enhancing image... {i}%")
                # Simulate enhancement step (can be replaced with actual enhancement)
                if i == 100:
                    output, _ = upsampler.enhance(img_np)
                    enhanced_img = Image.fromarray(output)
                    st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)

                    # Save enhanced image for download
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as out_file:
                        enhanced_img.save(out_file.name)
                        st.download_button("⬇️ Download", data=open(out_file.name, "rb"), file_name="enhanced.png")
                    break
        except Exception as e:
            st.error(f"❌ Failed to enhance image: {e}")
elif uploaded_image and not upsampler:
    st.warning("⚠️ Please load a model before enhancing images.")

# Feature: Model Info Tooltip
st.sidebar.markdown("""
    **Model Info:**  
    After uploading a model, you will see its architecture detected based on the `.pth` keys. Supported architectures are **RRDBNet** and **SRVGGNet**.
    If you see an "Unknown architecture" error, make sure the correct `.pth` model is uploaded and compatible with the RealESRGAN architecture.
""")
