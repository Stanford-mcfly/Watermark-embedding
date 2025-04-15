# Watermark Embedding Streamlit Application

This project is a Streamlit application that allows users to upload an image and watermark text, processes them in the frequency domain, and merges them to create a watermarked image. Additionally, it provides functionality to reveal the watermark from the watermarked image.

## Project Structure

```
streamlit-watermark-app
├── src
│   ├── app.py                # Main entry point of the Streamlit application
│   ├── utils
│   │   └── image_processing.py # Utility functions for image processing
├── requirements.txt          # Project dependencies
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-watermark-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501` to access the application.

3. Upload an image and enter the watermark text. The application will process the inputs and display the watermarked image.

4. Use the reveal feature to extract the watermark from the watermarked image.

## Application Functionality

- **Image Upload**: Users can upload an image file.
- **Watermark Text Input**: Users can input the text they want to use as a watermark.
- **Frequency Domain Processing**: The application converts the image and watermark text to the frequency domain for merging.
- **Watermarking**: The application merges the watermark with the image in the frequency domain.
- **Reveal Watermark**: Users can reveal the watermark from the watermarked image.

## License

This project is licensed under the MIT License.