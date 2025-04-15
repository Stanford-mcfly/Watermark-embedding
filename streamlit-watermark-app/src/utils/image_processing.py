import cv2
import numpy as np

def embed_logo_lsb(image, logo):
    """
    Embeds the binary representation of the logo into the least significant bits of the image.
    """
    # Resize the logo to match the dimensions of the image
    rows, cols = image.shape[:2]
    logo_resized = cv2.resize(logo, (cols, rows), interpolation=cv2.INTER_AREA)

    # Convert the logo to binary (0 or 1)
    _, binary_logo = cv2.threshold(logo_resized, 127, 1, cv2.THRESH_BINARY)

    # Ensure binary_logo is of type uint8
    binary_logo = binary_logo.astype(np.uint8)

    # Embed the binary logo into the least significant bit of the image
    watermarked_image = np.bitwise_and(image, 254)  # Clear LSB
    watermarked_image = np.bitwise_or(watermarked_image, binary_logo)  # Set LSB to logo bit
    return watermarked_image

def extract_logo_lsb(image):
    """
    Extracts the binary logo from the least significant bits of the image.
    """
    # Extract the least significant bit
    extracted_logo = np.bitwise_and(image, 1)  # Extract LSB
    extracted_logo = (extracted_logo * 255).astype(np.uint8)  # Scale back to 0-255 for visualization
    return extracted_logo