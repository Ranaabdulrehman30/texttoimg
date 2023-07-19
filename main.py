import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='SagemakerFullAcess-role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

from distutils.dir_util import copy_tree
from pathlib import Path
from huggingface_hub import snapshot_download
import random

HF_MODEL_ID="CompVis/stable-diffusion-v1-4"
HF_TOKEN="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" # your hf token: https://huggingface.co/settings/tokens
assert len(HF_TOKEN) > 0, "Please set HF_TOKEN to your huggingface token. You can find it here: https://huggingface.co/settings/tokens"

# download snapshot
snapshot_dir = snapshot_download(repo_id=HF_MODEL_ID,revision="fp16",use_auth_token=HF_TOKEN)

# create model dir
model_tar = Path(f"model-{random.getrandbits(16)}")
model_tar.mkdir(exist_ok=True)

# copy snapshot to model dir
copy_tree(snapshot_dir, str(model_tar))

# copy code/ to model dir
copy_tree("code/", str(model_tar.joinpath("code")))

import tarfile
import os

# helper to create the model.tar.gz
def compress(tar_dir=None,output_file="model.tar.gz"):
    parent_dir=os.getcwd()
    os.chdir(tar_dir)
    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir('.'):
          print(item)
          tar.add(item, arcname=item)
    os.chdir(parent_dir)

compress(str(model_tar))

from sagemaker.s3 import S3Uploader

# upload model.tar.gz to s3
s3_model_uri=S3Uploader.upload(local_path="model.tar.gz", desired_s3_uri=f"s3://{sess.default_bucket()}/stable-diffusion-v1-4")

print(f"model uploaded to: {s3_model_uri}")


from sagemaker.huggingface.model import HuggingFaceModel


# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data=s3_model_uri,      # path to your model and script
   role=role,                    # iam role with permissions to create an Endpoint
   transformers_version="4.17",  # transformers version used
   pytorch_version="1.10",       # pytorch version used
   py_version='py38',            # python version used
)

# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
    )

from PIL import Image
from io import BytesIO
from IPython.display import display
import base64
import matplotlib.pyplot as plt

# helper decoder
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
    plt.show()  


num_images_per_prompt = 3
prompt = "A dog trying catch a flying pizza art drawn by disney concept artists, golden colour, high quality, highly detailed, elegant, sharp focus"

# run prediction
response = predictor.predict(data={
  "inputs": prompt,
  "num_images_per_prompt" : num_images_per_prompt
  }
)

# decode images
decoded_images = [decode_base64_image(image) for image in response["generated_images"]]

# visualize generation
display_images(decoded_images)


