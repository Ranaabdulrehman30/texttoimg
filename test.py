import boto3
import json
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# Helper decoder
def decode_base64_image(image_string):
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)

# Display PIL images as grid
def display_images(images=None, columns=3, width=100, height=100):
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.axis('off')
        plt.imshow(image)
    plt.show()  # Display the plot

prompt = "Formula 1 car with old school design."

request_body = {
    "inputs": prompt,
    "num_images_per_prompt": 3
}

# Serialize data for endpoint
payload = json.dumps(request_body)
client = boto3.client("sagemaker-runtime")
response = client.invoke_endpoint(
    # Change to your endpoint name returned in the previous step
    EndpointName="huggingface-pytorch-inference-2023-07-17-19-10-20-472",
    ContentType="application/json",
    Body=payload
)

# Decode images
response_data = json.loads(response["Body"].read().decode('utf-8'))
image_data = response_data["generated_images"]
decoded_images = [decode_base64_image(image) for image in image_data]

# Visualize generation
display_images(decoded_images)
