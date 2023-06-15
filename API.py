from flask import Flask, request, jsonify
import torch
import torchvision
from PIL import Image, ImageDraw
import io

app = Flask(__name__)

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains an image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found.'}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')

    # Perform the prediction
    results = model(image)

    # Access the predicted bounding boxes and labels
    predictions = results.pandas().xyxy[0]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for _, prediction in predictions.iterrows():
        label = prediction['name']
        confidence = prediction['confidence']
        xmin, ymin, xmax, ymax = prediction['xmin'], prediction['ymin'], prediction['xmax'], prediction['ymax']

        # Draw the bounding box rectangle
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)

        # Add label and confidence text
        text = f'{label} {confidence:.2f}'
        draw.text((xmin, ymin - 15), text, fill='red')

    # Save the image with bounding boxes to a byte stream
    image_stream = io.BytesIO()
    image.save(image_stream, format='JPEG')
    image_stream.seek(0)

    # Return the image file with bounding boxes
    return app.response_class(image_stream, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
