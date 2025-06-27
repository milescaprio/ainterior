import requests
import base64


# Load and encode image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return f.read()


# Define your parameters
image_path = "bus.jpg"  # Replace with your local image file path
from keys import REPLICATE_API_TOKEN as api_token  # Replace with your actual token

# Prepare payload
endpoint_url = "https://api.replicate.com/v1/predictions"
headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}


# Upload the image to Replicate's file hosting
def upload_to_replicate(image_path, api_token):
    with open(image_path, "rb") as f:
        response = requests.post(
            "https://dreambooth-api-experimental.replicate.com/v1/upload",
            headers={"Authorization": f"Token {api_token}"},
            files={"file": f},
        )
    response.raise_for_status()
    return response.json()["upload_url"], response.json()["serving_url"]


# Upload the image and get the URL
upload_url, image_url = upload_to_replicate(image_path, api_token)

# Modify these parameters based on jagilley/controlnet input schema
input_data = {
    "image": image_url,
    "prompt": "A fantasy castle on a hill",
    "controlnet_conditioning_scale": 1.0,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    # Add other params like "model", "controlnet_type", etc. as needed
}

# Send the request to Replicate
response = requests.post(
    endpoint_url,
    headers=headers,
    json={
        "version": "jagilley/controlnet:8ebda4c70b3ea2a2bf86e44595afb562a2cdf85525c620f1671a78113c9f325b",  # Get the latest version from the model page
        "input": input_data,
    },
)

# Print result
if response.ok:
    print(response.json())
else:
    print("Error:", response.text)
