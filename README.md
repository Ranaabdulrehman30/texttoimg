
![Screenshot (54)](https://github.com/Ranaabdulrehman30/texttoimg/assets/96223013/f37e1653-4aa8-4409-a87b-0da9fffb4dc7)

## 1. Create Stable Diffusion inference script
Amazon SageMaker allows us to customize the inference script by providing a inference.py file. The inference.py file is the entry point to our model. It is responsible for loading the model and handling the inference request. If you are used to deploying Hugging Face Transformers that might be knew to you. Usually, we just provide the HF_MODEL_ID and HF_TASK and the Hugging Face DLC takes care of the rest. For diffusers thats not yet possible. We have to provide the inference.py file and implement the model_fn and predict_fn functions.

In addition to the inference.py file we also have to provide a requirements.txt file. The requirements.txt file is used to install the dependencies for our inference.py file.

The first step is to create a code/ directory.

The last step for our inference handler is to create the inference.py file. The inference.py file is responsible for loading the model and handling the inference request. The model_fn function is called when the model is loaded. The predict_fn function is called when we want to do inference.

We are using the diffusers library to load the model in the model_fn and generate 4 image for an input prompt with the predict_fn. The predict_fn function returns the 4 image as a base64 encoded string.

## 2. Create SageMaker model.tar.gz artifact

To use our inference.py we need to bundle it together with our model weights into a model.tar.gz. The archive includes all our model-artifcats to run inference. The inference.py script will be placed into a code/ folder. We will use the huggingface_hub SDK to easily download CompVis/stable-diffusion-v1-4 from Hugging Face and then upload it to Amazon S3 with the sagemaker SDK.

Before we can load our model from the Hugging Face Hub we have to make sure that we accepted the license of CompVis/stable-diffusion-v1-4 to be able to use it. CompVis/stable-diffusion-v1-4 is published under the CreativeML OpenRAIL-M license. You can accept the license by clicking on the Agree and access repository button on the model page at: https://huggingface.co/CompVis/stable-diffusion-v1-4.

Before we can load the model make sure you have a valid HF Token. You can create a token by going to your Hugging Face Settings and clicking on the New token button. Make sure the enviornment has enough diskspace to store the model, ~30GB should be enough.

The next step is to copy the code/ directory into the model/ directory.

### copy code/ to model dir
copy_tree("code/", str(model_tar.joinpath("code")))


After we created the model.tar.gz archive we can upload it to Amazon S3. We will use the sagemaker SDK to upload the model to our sagemaker session bucket.

## 3. Deploy the model to Amazon SageMaker
After we have uploaded our model archive we can deploy our model to Amazon SageMaker. We will use HuggingfaceModel to create our real-time inference endpoint.

We are going to deploy the model to an g4dn.xlarge instance. The g4dn.xlarge instance is a GPU instance with 1 NVIDIA Tesla T4 GPUs. If you are interested in how you could add autoscaling to your endpoint you can check out Going Production: Auto-scaling Hugging Face Transformers with Amazon SageMaker.

## 4. Generate images using the deployed model

The .deploy() returns an HuggingFacePredictor object which can be used to request inference. Our endpoint expects a json with at least inputs key. The inputs key is the input prompt for the model, which will be used to generate the image. Additionally, we can provide num_inference_steps, guidance_scale & num_images_per_prompt to controll the generation.

The predictor.predict() function returns a json with the generated_images key. The generated_images key contains the 4 generated images as a base64 encoded string. To decode our response we added a small helper function decode_base64_to_image which takes the base64 encoded string and returns a PIL.Image object and display_images, which takes a list of PIL.Image objects and displays them.

Now, lets generate some images. As example lets generate 3 images for the prompt A dog trying catch a flying pizza art drawn by disney concept artists. Generating 3 images takes around 30 seconds.

## 4. Lambda Funtion

After model deployement and it's testing locally. Let integrate the Endpoint name in the Lambda Script, along with the customized script to upload the images in the s3 bucket. The Lambda Fucntion takes as prompt and returns the URL of the images generated. The s3 bucket itself is private so the these URLs are not directly accessible. 
The lambda fucntion is then deployed on aws with the name (texttoimg) along with the addition of required layers.

## 5. Here are the steps to add more models to an existing endpoint:

1. Create new models: Prepare and train the additional models you want to add to the endpoint. Each model should be saved as a tarball file (model.tar.gz) and uploaded to an S3 bucket.

2. Register the new models: Use the AWS CLI or SageMaker API to register the new models. For each model, provide a unique model name and the S3 location of the model artifacts (model.tar.gz).

   
   (aws sagemaker create-model --model-name <new_model_name> --primary-container Image<container_image>,ModelDataUrl=s3://<bucket_name>/<model_tarball_location>)
   
   Replace <new_model_name> with the desired name for the new model, <container_image> with the appropriate container image for the model, <bucket_name> with the name of the S3 bucket where the model artifacts are stored, and <model_tarball_location> with the location of the model.tar.gz file within the bucket.

   Repeat this step for each additional model you want to add.
3. Update the endpoint: Once the new models are registered, you can update the endpoint to include them. Use the AWS CLI or the SageMaker API to update the endpoint with the new models.

   (aws sagemaker update-endpoint --endpoint-name <endpoint_name> --desired-capacity <new_desired_capacity> --production-variants "[{ModelName=<existing_model_name>,VariantName=<variant_name>,InitialInstanceCount=<instance_count>,InstanceType=<instance_type>},{ModelName=<new_model_name>,VariantName=<variant_name>,InitialInstanceCount=<instance_count>,InstanceType=<instance_type>}]")

   Replace <endpoint_name> with the name of your existing endpoint, <new_desired_capacity> with the desired number of instances for the updated endpoint, <existing_model_name> with the name of the model already associated with the endpoint, <variant_name> with the name for the variant (e.g., "variant1"), <instance_count> with the desired number of instances for each model, and <instance_type> with the desired instance type.

   Repeat the {ModelName=...,VariantName=...,InitialInstanceCount=...,InstanceType=...} section for each model you want to add, specifying the 
    corresponding details.
4. Wait for the endpoint update: The endpoint update process may take some time. You can monitor the status of the endpoint using the AWS Management Console or the AWS CLI:
    (aws sagemaker describe-endpoint --endpoint-name <endpoint_name> --query "EndpointStatus")

   Wait until the endpoint status changes to "InService" before proceeding.

After following these steps, your SageMaker endpoint will be serving multiple models concurrently. Each model can be invoked separately by specifying the model name or variant name in the inference requests.

## 6. To tweak the parameters of the Hugging Face Stable Diffusion model deployed on an AWS SageMaker endpoint, you'll need to follow these general steps:
1. Modify the model parameters: Open the model configuration file in a text editor. : Look for a file named "config.json" or similar within the model's directory structure. Look for the relevant sections or variables in the configuration file that correspond to the parameters you want to tweak. Modify the parameter values according to your requirements.
2. Save the modified model configuration: After making the desired parameter changes, save the modified model configuration file.
3. Upload the modified model configuration to S3: Use the AWS CLI to upload the modified model configuration file back to the S3 bucket: (aws s3 cp <local_modified_config_file_path> s3://<bucket_name>/<config_file_path>)

   Replace <local_modified_config_file_path> with the local file path of the modified configuration file, <bucket_name> with the name of the S3 bucket, and <config_file_path> with the desired path for the configuration file within the bucket.

4. Update the SageMaker endpoint: Finally, update the SageMaker endpoint to use the modified model configuration. Use the AWS CLI or the SageMaker API to update the endpoint configuration with the new model configuration: (aws sagemaker update-endpoint --endpoint-name <endpoint_name> --endpoint-config-name <endpoint_config_name>)

   Replace <endpoint_name> with the name of your existing SageMaker endpoint, and <endpoint_config_name> with the name of the updated endpoint configuration.

By following these steps, you can tweak the Stable Diffusion model parameters by modifying the model configuration file and updating the SageMaker endpoint to use the modified configuration.





## Delete model and endpoint
To clean up, we can delete the model and endpoint.

predictor.delete_model()

predictor.delete_endpoint()

