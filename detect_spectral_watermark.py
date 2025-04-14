import cv2
import numpy as np

def detect_spectral_watermark(image_path, original_path=None):
    # Load the watermarked image
    watermarked_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if watermarked_img is None:
        print("Error: Unable to load the watermarked image.")
        return

    # Perform DFT on the watermarked image
    dft_watermarked = cv2.dft(np.float32(watermarked_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_watermarked_shift = np.fft.fftshift(dft_watermarked)

    if original_path:
        # Load the original image
        original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print("Error: Unable to load the original image.")
            return

        # Perform DFT on the original image
        dft_original = cv2.dft(np.float32(original_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_original_shift = np.fft.fftshift(dft_original)

        # Subtract the original spectrum from the watermarked spectrum
        diff = cv2.magnitude(dft_watermarked_shift[:, :, 0] - dft_original_shift[:, :, 0],
                             dft_watermarked_shift[:, :, 1] - dft_original_shift[:, :, 1])
    else:
        # Analyze the magnitude spectrum of the watermarked image
        diff = cv2.magnitude(dft_watermarked_shift[:, :, 0], dft_watermarked_shift[:, :, 1])

    # Normalize the difference for visualization
    diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to detect anomalies (potential watermark regions)
    _, thresholded = cv2.threshold(diff_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Count the number of significant anomalies
    anomalies = np.sum(thresholded > 0)

    # Display results
    if anomalies > 0:
        print("Spectral watermark detected in the image.")
    else:
        print("No spectral watermark detected in the image.")

    # Save the difference image for inspection
    cv2.imwrite("spectral_watermark_diff.png", diff_normalized)
    print("Difference image saved as spectral_watermark_diff.png")


# Example usage
if __name__ == "__main__":
    detect_spectral_watermark("spectral_watermarked.png", "input.jpg")
