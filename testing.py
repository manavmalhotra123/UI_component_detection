import requests

# Set the API endpoint URL
api_url = 'http://localhost:5000/predict'

# Set the path to the image file
image_path = 'test.jpeg'

# Send a POST request to the API with the image file
with open(image_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(api_url, files=files)

# Check if the request was successful
if response.status_code == 200:
    # Save the image file with bounding boxes
    output_image_path = 'output_image.jpeg'
    with open(output_image_path, 'wb') as output_file:
        output_file.write(response.content)
    
    print(f'Image with bounding boxes saved to: {output_image_path}')
else:
    print('Error:', response.text)
