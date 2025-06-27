import requests
import base64
import mimetypes
import os

# --- CONFIGURATION ---
image_path = "bus.jpg"  # Path to your image
from keys import REPLICATE_API_TOKEN as api_token  # Replace with your actual token

model_version = "jagilley/controlnet:8ebda4c70b3ea2a2bf86e44595afb562a2cdf85525c620f1671a78113c9f325b"  # Replace with actual model version


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


import time


def wait_for_prediction(prediction_url, headers, timeout=60):
    start_time = time.time()
    while True:
        response = requests.get(prediction_url, headers=headers)
        if not response.ok:
            raise RuntimeError(f"Error fetching prediction: {response.text}")

        prediction = response.json()
        status = prediction["status"]
        print(f"Status: {status}")

        if status == "succeeded":
            print("✅ Prediction complete!")
            print("Output:", prediction["output"])
            return prediction["output"]
        elif status == "failed":
            raise RuntimeError(f"❌ Prediction failed: {prediction}")

        if time.time() - start_time > timeout:
            raise TimeoutError("Prediction took too long.")

        time.sleep(2)  # Poll every 2 seconds


# After POST request
prediction = response.json()
prediction_url = prediction["urls"]["get"]
output = wait_for_prediction(prediction_url, headers)
