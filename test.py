import torch
import torchvision
from PIL import Image, ImageDraw

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')

# Set the model to inference mode
model.eval()

# Load an image
image_path = 'test.jpeg'
image = Image.open(image_path)

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

# Save the image with bounding boxes
image_with_boxes_path = 'test_with_boxes.jpeg'
image.save(image_with_boxes_path)

# Display the path to the image with bounding boxes
print(f'Image with bounding boxes saved to: {image_with_boxes_path}')
