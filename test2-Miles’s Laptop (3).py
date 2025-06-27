import requests
import base64
import mimetypes
import os

# --- CONFIGURATION ---
image_path = "your_image.jpg"  # Path to your image
api_token = "your_replicate_api_token"  # Replace with your Replicate token
model_version = "INSERT_MODEL_VERSION_HERE"  # Replace with actual model version


# --- ENCODE IMAGE AS DATA URI ---
def encode_image_to_data_uri(path):
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        raise ValueError("Could not determine MIME type.")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


data_uri = encode_image_to_data_uri(image_path)

# --- PREPARE REQUEST ---
endpoint = "https://api.replicate.com/v1/predictions"
headers = {
    "Authorization": f"Token {api_token}",
    "Content-Type": "application/json",
}

payload = {
    "version": model_version,
    "input": {
        "image": data_uri,
        "prompt": "A futuristic cityscape",
        "controlnet_conditioning_scale": 1.0,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        # Add other fields as needed
    },
}

# --- SEND REQUEST ---
response = requests.post(endpoint, headers=headers, json=payload)

# --- HANDLE RESPONSE ---
if response.ok:
    prediction = response.json()
    print("Prediction ID:", prediction["id"])
    print("Status:", prediction["status"])
else:
    print("Error:", response.status_code, response.text)
