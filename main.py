import cv2
import numpy as np

def embed_watermark_color(image_path, watermark_text, output_path='watermarked_color.png'):
    # Load color image
    img = cv2.imread(image_path)
    b, g, r = cv2.split(img)
    channels = [b, g, r]
    watermarked_channels = []

    for channel in channels:
        rows, cols = channel.shape

        # DFT of the channel
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Create multiple small watermarks scattered across the image
        watermark = np.zeros((rows, cols), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        step = 100  # Distance between watermarks
        for i in range(10, rows, step):
            for j in range(10, cols, step):
                cv2.putText(watermark, watermark_text, (j, i), font, 0.5, 255, 1, cv2.LINE_AA)

        # DFT of watermark
        watermark_dft = cv2.dft(np.float32(watermark), flags=cv2.DFT_COMPLEX_OUTPUT)
        watermark_dft_shift = np.fft.fftshift(watermark_dft)

        # Embed watermark
        alpha = 0.05  # Reduced watermark strength
        dft_shift += alpha * watermark_dft_shift

        # Inverse DFT
        dft_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(dft_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        watermarked_channels.append(img_back)

    # Merge channels back
    watermarked_img = cv2.merge(watermarked_channels)
    cv2.imwrite(output_path, watermarked_img)
    print(f"Watermarked color image saved as {output_path}")


def extract_watermark_color(watermarked_path, original_path, output_path='extracted_watermark_color.png'):
    # Load both images
    watermarked = cv2.imread(watermarked_path)
    original = cv2.imread(original_path)

    # Split channels
    b_w, g_w, r_w = cv2.split(watermarked)
    b_o, g_o, r_o = cv2.split(original)

    extracted = []

    for cw, co in zip([b_w, g_w, r_w], [b_o, g_o, r_o]):
        # Get DFTs
        dft_w = np.fft.fftshift(cv2.dft(np.float32(cw), flags=cv2.DFT_COMPLEX_OUTPUT))
        dft_o = np.fft.fftshift(cv2.dft(np.float32(co), flags=cv2.DFT_COMPLEX_OUTPUT))

        # Subtract to get watermark
        diff = dft_w - dft_o

        # Inverse DFT
        ishift = np.fft.ifftshift(diff)
        watermark = cv2.idft(ishift)
        watermark = cv2.magnitude(watermark[:, :, 0], watermark[:, :, 1])
        watermark = cv2.normalize(watermark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        extracted.append(watermark)

    # Merge watermarks or just use one (e.g., blue channel)
    final_extracted = extracted[0]  # You can average or blend if needed
    cv2.imwrite(output_path, final_extracted)
    print(f"Extracted watermark saved as {output_path}")


def detect_watermark(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Unable to load the image.")
        return

    # Perform DFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Compute the magnitude spectrum
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Analyze the spectrum for anomalies (e.g., peaks)
    mean_val = np.mean(magnitude_spectrum)
    std_dev = np.std(magnitude_spectrum)

    # Threshold for detecting anomalies
    threshold = mean_val + 3 * std_dev
    anomalies = np.sum(magnitude_spectrum > threshold)

    # Check for structured patterns in the frequency domain
    if anomalies > 0:
        print("Watermark detected in the image.")
    else:
        print("No watermark detected in the image.")


# Example usage
if __name__ == "__main__":
    # Embed multiple watermarks
    embed_watermark_color("input.jpg", "HiddenMark")

    # Detect watermark
    detect_watermark("input.jpg")
