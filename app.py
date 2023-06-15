import streamlit as st
import requests
from PIL import Image
import io

# Set the API endpoint
API_ENDPOINT = 'http://localhost:5000/predict'

# Define the Streamlit app
def main():
    st.title('YOLOv5 Object Detection')

    # Upload image file
    image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        # Display uploaded image
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert image to bytes-like object
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # Make prediction request to the Flask API
        response = requests.post(API_ENDPOINT, files={'image': image_bytes})

        # Display the image with bounding boxes
        if response.status_code == 200:
            st.subheader('Detected Objects')
            image_with_boxes = Image.open(io.BytesIO(response.content))
            st.image(image_with_boxes, caption='Objects Detected', use_column_width=True)
        else:
            st.error('Error occurred during prediction.')

# Run the app
if __name__ == '__main__':
    main()
