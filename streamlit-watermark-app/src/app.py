import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from utils.image_processing import (
    embed_logo_lsb,
    extract_logo_lsb
)

def main():
    st.title("LSB Watermark Embedding and Detection App")

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Embed Logo Watermark", "Detect Logo Watermark"])

    # Tab 1: Embed Logo Watermark
    with tab1:
        st.header("Embed Logo Watermark")
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        uploaded_logo = st.file_uploader("Upload a Logo", type=["jpg", "jpeg", "png"])

        if uploaded_image and uploaded_logo:
            # Convert uploaded image and logo to arrays
            image = Image.open(uploaded_image)
            image_array = np.array(image)

            logo = Image.open(uploaded_logo).convert("L")  # Convert logo to grayscale
            logo_array = np.array(logo)

            if len(image_array.shape) == 3:  # Colored image
                # Process each channel separately
                channels = cv2.split(image_array)
                watermarked_channels = [embed_logo_lsb(channel, logo_array) for channel in channels]
                watermarked_image = cv2.merge(watermarked_channels)
            else:  # Grayscale image
                watermarked_image = embed_logo_lsb(image_array, logo_array)

            # Display the watermarked image
            st.image(watermarked_image, caption="Watermarked Image", use_column_width=True)

            # Add download button for watermarked image
            watermarked_pil = Image.fromarray(watermarked_image)
            buf = BytesIO()
            watermarked_pil.save(buf, format="PNG")
            byte_data = buf.getvalue()
            st.download_button(
                label="Download Watermarked Image",
                data=byte_data,
                file_name="watermarked_image.png",
                mime="image/png"
            )

    # Tab 2: Detect Logo Watermark
    with tab2:
        st.header("Detect Logo Watermark")
        uploaded_watermarked_image = st.file_uploader("Upload a Watermarked Image", type=["jpg", "jpeg", "png"])

        if uploaded_watermarked_image:
            # Convert uploaded image to array
            watermarked_image = Image.open(uploaded_watermarked_image)
            watermarked_array = np.array(watermarked_image)

            if len(watermarked_array.shape) == 3:  # Colored image
                # Process each channel separately
                channels = cv2.split(watermarked_array)
                extracted_channels = [extract_logo_lsb(channel) for channel in channels]
                extracted_logo = cv2.merge(extracted_channels)
            else:  # Grayscale image
                extracted_logo = extract_logo_lsb(watermarked_array)

            # Display the extracted logo
            st.image(extracted_logo, caption="Extracted Logo", use_column_width=True)

if __name__ == "__main__":
    main()