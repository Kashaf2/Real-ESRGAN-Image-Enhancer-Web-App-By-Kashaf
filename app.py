import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import tempfile
from io import BytesIO
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

st.set_page_config(page_title="Real-ESRGAN Image Enhancer", layout="centered")
st.title("🖼️ Real-ESRGAN Image Enhancer")
st.markdown("Enhance images with AI Super Resolution. Designed by **Kashaf Bin Shamim**.")

# Sidebar: Upload model
st.sidebar.header("1. Upload RealESRGAN Model (.pth)")
model_file = st.sidebar.file_uploader("Upload model weights", type=["pth"])

# Sidebar: Select scale factor
scale = st.sidebar.selectbox("2. Choose Upscale Factor", [2, 4], index=1)

# Optional Features
face_enhance = st.sidebar.checkbox("Enable Face Enhancement (GFPGAN)", value=False)
jpeg_remove = st.sidebar.checkbox("Remove JPEG Artifacts", value=False)

# Function to detect model architecture from .pth file
def detect_model_architecture(model_path):
    try:
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "params_ema" in ckpt:
                ckpt = ckpt["params_ema"]
            elif "params" in ckpt:
                ckpt = ckpt["params"]

        keys = list(ckpt.keys())
        if any("body.0" in key for key in keys):
            return "RRDBNet"
        elif any("conv_first" in key or "body.0.conv1" in key for key in keys):
            return "SRVGGNetCompact"
        else:
            return None
    except Exception as e:
        return None

# Get architecture instance
def get_model(arch_name, scale):
    if arch_name == "RRDBNet":
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=scale)
    elif arch_name == "SRVGGNetCompact":
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                               num_feat=64, num_conv=32, upscale=scale, act_type="prelu")
    else:
        return None

# Load model if uploaded
upsampler = None
gfpgan_enhancer = None
confirmed_arch = None

if model_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        tmp.write(model_file.read())
        tmp_model_path = tmp.name

    arch_name = detect_model_architecture(tmp_model_path)
    
    if not arch_name:
        st.sidebar.error("❌ Unknown or unsupported model architecture.")
    else:
        st.sidebar.success(f"✅ Detected architecture: {arch_name}")
        model = get_model(arch_name, scale)
        
        if model is None:
            st.sidebar.error("⚠️ Architecture supported but not implemented.")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                # Load model on the selected device (GPU if available)
                model.load_state_dict(torch.load(tmp_model_path, map_location=device), strict=False)
                
                # Initialize upsampler with GPU support
                upsampler = RealESRGANer(
                    scale=scale,
                    model_path=tmp_model_path,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=(device.type == "cuda"),
                    device=device
                )
                
                # If face enhancement is enabled, set up the face enhancer with the same device
                if face_enhance:
                    gfpgan_enhancer = GFPGANer(
                        model_path=None,
                        upscale=scale,
                        arch="clean",
                        channel_multiplier=2,
                        bg_upsampler=upsampler
                    )
                st.sidebar.success("✅ Model loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load model: {e}")

# Upload image
st.header("3. Upload Image to Enhance")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image and upsampler:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Original Image", use_column_width=True)

    progress_bar = st.progress(0)
    progress_text = st.empty()

    if st.button("🚀 Enhance Image"):
        try:
            img_np = np.array(img)
            for i in range(1, 101):
                progress_bar.progress(i)
                progress_text.text(f"Enhancing image... {i}%")

            if face_enhance and gfpgan_enhancer:
                _, _, enhanced_img = gfpgan_enhancer.enhance(img_np, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img_np, outscale=scale)
                enhanced_img = Image.fromarray(output)

            st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as out_file:
                enhanced_img.save(out_file.name)
                st.download_button("⬇️ Download Enhanced Image", data=open(out_file.name, "rb"),
                                   file_name="enhanced.png", mime="image/png")

        except Exception as e:
            st.error(f"❌ Failed to enhance image: {e}")
elif uploaded_image and not upsampler:
    st.warning("⚠️ Please upload and confirm a model first.")

# Footer
st.markdown("---")
st.markdown(
    "<center><sub>Designed by <strong>Kashaf Bin Shamim</strong></sub></center>",
    unsafe_allow_html=True
)
