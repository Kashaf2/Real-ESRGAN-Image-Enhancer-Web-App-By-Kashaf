# Real-ESRGAN Web UI

This is a web-based application that allows users to upscale images using Real-ESRGAN. The app is built using Streamlit and provides an easy-to-use interface for enhancing image resolution. Users can upload models, select upscale factors, and upload images for enhancement.

## Features
- **Upload Custom Models**: Users can upload `.pth` files to use custom Real-ESRGAN models for image enhancement.
- **Choose Upscale Factor**: Select an upscale factor of 2x or 4x.
- **Enhance Images**: Upload an image and enhance it with the selected model and upscale factor.
- **Download Enhanced Image**: After enhancement, download the high-resolution image.

## Installation

### 1. Clone this repository
Clone the repository to your local machine:

```bash
git clone https://github.com/Kashaf2/Real-ESRGAN-Web-UI.git

2. Install dependencies
Navigate to the project directory and install the required dependencies:

cd Real-ESRGAN-Web-UI
pip install -r requirements.txt

3. Install Real-ESRGAN
To install Real-ESRGAN, follow the instructions from the Real-ESRGAN repository if not already included in the dependencies.

4. Run the app
After installing the dependencies, you can start the Streamlit app:
streamlit run app.py

5. Upload Model and Images
Step 1: In the sidebar, upload a .pth model file that you wish to use.

Step 2: Choose the upscale factor (2x or 4x).

Step 3: Upload an image you want to upscale.

Step 4: Click on the "Enhance Image" button to process the image.

Once processed, the enhanced image will be displayed, and you can download it by clicking the download button.

Contributing
Feel free to fork the repository, create a pull request, or open an issue if you encounter any problems.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Real-ESRGAN

Streamlit

Feel free to customize the sections as per your project needs.




