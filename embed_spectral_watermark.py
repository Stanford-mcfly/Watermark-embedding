import cv2
import numpy as np

def embed_spectral_watermark(image_path, watermark_text, output_path='spectral_watermarked.png'):
    # Load image (color or grayscale)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load the image.")
        return

    if len(img.shape) == 2:  # Grayscale image
        channels = [img]
    else:  # Color image
        channels = cv2.split(img)

    watermarked_channels = []

    for channel in channels:
        rows, cols = channel.shape

        # Perform DFT
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create watermark in the frequency domain
        watermark = np.zeros((rows, cols), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        step = 100  # Distance between watermarks
        for i in range(10, rows, step):
            for j in range(10, cols, step):
                cv2.putText(watermark, watermark_text, (j, i), font, 0.3, 255, 1, cv2.LINE_AA)

        # DFT of the watermark
        watermark_dft = cv2.dft(np.float32(watermark), flags=cv2.DFT_COMPLEX_OUTPUT)
        watermark_dft_shift = np.fft.fftshift(watermark_dft)

        # Embed watermark in the spectral domain
        alpha = 0.02  # Reduced watermark strength for less visibility
        dft_shift += alpha * watermark_dft_shift

        # Inverse DFT
        dft_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(dft_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        watermarked_channels.append(img_back)

    # Merge channels back for color images
    if len(watermarked_channels) > 1:
        watermarked_img = cv2.merge(watermarked_channels)
    else:
        watermarked_img = watermarked_channels[0]

    # Save the watermarked image
    cv2.imwrite(output_path, watermarked_img)
    print(f"Spectral watermarked image saved as {output_path}")


# Example usage
if __name__ == "__main__":
    embed_spectral_watermark("input.jpg", "SpectralMark")
