import boto3
import json
from PIL import Image
from io import BytesIO
import base64 
#import matplotlib.pyplot as plt

# Helper decoder
def decode_base64_image(image_string):
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


def lambda_handler(event, context):
    # Get the prompt from the request payload
    request_body = json.loads(event["body"])
    prompt = request_body["prompt"]
 
    # Send the prompt to the SageMaker endpoint for image generation
    request_data = {
        "inputs": prompt,
        "num_images_per_prompt": 3
    }
    payload = json.dumps(request_data)
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName="huggingface-pytorch-inference-2023-07-17-19-10-20-472",
        ContentType="application/json",
        Body=payload 
    )

    # Decode images
    response_data = json.loads(response["Body"].read().decode('utf-8'))
    image_data = response_data["generated_images"]
    decoded_images = [decode_base64_image(image) for image in image_data]

    # Upload images to S3 bucket
    s3_client = boto3.client("s3")
    s3_bucket_name = "texttoimg-bucket"

    s3_image_urls = []
    for i, image in enumerate(decoded_images):
        image_key = f"generated_image_{i}.png"
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        s3_client.upload_fileobj(buffer, s3_bucket_name, image_key)

        # Construct the S3 URL
        s3_image_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{image_key}"
        s3_image_urls.append(s3_image_url)

    # Return the S3 URLs in the API response
    response_payload = {
        "image_urls": s3_image_urls
    }

    return {
        "statusCode": 200,
        "body": json.dumps(response_payload)
    }