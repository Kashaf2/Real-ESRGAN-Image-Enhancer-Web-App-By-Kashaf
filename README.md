Real-ESRGAN Image Enhancer Web App
This project is a Real-ESRGAN Image Enhancer built using Streamlit, allowing users to upload images and enhance them with the power of Real-ESRGAN upscaling models. The web app allows users to:

Upload Real-ESRGAN Model (.pth): Select from supported Real-ESRGAN models, including RRDBNet and SRVGGNet, or upload custom models.

Select Upscale Factor: Choose the upscale factor (2x or 4x) for image enhancement.

Enhance Images: Upload an image and enhance it by increasing its resolution using the selected Real-ESRGAN model.

Download Enhanced Images: Get the enhanced images for download in high-definition quality.

The app supports automatic architecture detection from the .pth model file, streamlining the user experience.

Key Features:
Model Upload: Upload any compatible Real-ESRGAN .pth model.

Architecture Detection: Automatically detects the model architecture (e.g., RRDBNet, SRVGGNet) and loads the appropriate model.

Image Enhancement: Easily upscale images by 2x or 4x with the Real-ESRGAN upscaling method.

Streamlit Interface: Clean and simple interface for easy image enhancement.

Progress Bar: View the enhancement process in real-time.

Download Enhanced Image: After enhancement, users can download the upscaled image.

Installation:
To run this project locally, clone this repository and install the dependencies.

Clone the repository:

bash:
git clone https://github.com/your-username/real-esrgan-image-enhancer.git
Navigate to the project directory:

bash:
cd real-esrgan-image-enhancer
Install the required Python dependencies:

bash:
pip install -r requirements.txt

Run the Streamlit app:

bash:
streamlit run app.py

Supported Models:
RealESRGAN x4 (RRDBNet architecture)

RealESRGAN x2 (SRVGGNet architecture)

Custom Real-ESRGAN models can be uploaded as well!

Requirements:
Python 3.7+

Streamlit

PyTorch

RealESRGAN Pretrained Models

License:
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments:
Real-ESRGAN for the powerful image enhancement algorithm.

Streamlit for providing an intuitive interface for building the web app.
