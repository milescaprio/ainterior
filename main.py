import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import json
import requests
import os

# Suppress warnings from diffusers/transformers for cleaner output
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Enable MPS fallback for Apple Silicon if GPU memory is low
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress tokenizers warning

# --- Model Imports (Conditional to prevent immediate download for imports) ---
# We'll load these only when needed to save memory and time if not used
# and to manage dependencies better.
# For local development, uncomment these and manage model downloads carefully.
# from ultralytics import YOLO
# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from transformers import pipeline
# import torch

# --- Configuration ---
# Your Gemini API Key (replace with your actual key)
from keys import GEMINI_API_KEY  # Ensure you have a keys.py file with your API key

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# --- Mock Product Database ---
# A simplified database for product linking. In a real app, this would query
# external APIs or a robust internal database.
MOCK_PRODUCT_DATABASE = {
    "bed": [
        {
            "name": "Malm Bed Frame",
            "store": "IKEA",
            "mock_url": "https://example.com/ikea/malm_bed",
        },
        {
            "name": "Platform Bed",
            "store": "Target",
            "mock_url": "https://example.com/target/platform_bed",
        },
        {"name": "Generic Bed", "store": "Generic", "mock_url": "N/A"},
    ],
    "sofa": [
        {
            "name": "Ektorp Sofa",
            "store": "IKEA",
            "mock_url": "https://example.com/ikea/ektorp_sofa",
        },
        {
            "name": "Modular Sectional",
            "store": "Wayfair",
            "mock_url": "https://example.com/wayfair/sectional",
        },
        {"name": "Generic Sofa", "store": "Generic", "mock_url": "N/A"},
    ],
    "nightstand": [
        {
            "name": "Hemnes Nightstand",
            "store": "IKEA",
            "mock_url": "https://example.com/ikea/hemnes_nightstand",
        },
        {
            "name": "Round Side Table",
            "store": "Amazon",
            "mock_url": "https://example.com/amazon/round_table",
        },
        {"name": "Generic Nightstand", "store": "Generic", "mock_url": "N/A"},
    ],
    "art_piece": [
        {
            "name": "Abstract Canvas Print",
            "store": "Amazon",
            "mock_url": "https://example.com/amazon/abstract_art",
        },
        {
            "name": "Framed Poster",
            "store": "Target",
            "mock_url": "https://example.com/target/framed_poster",
        },
        {"name": "Generic Wall Art", "store": "Generic", "mock_url": "N/A"},
    ],
    "desk": [
        {
            "name": "Linnmon Desk",
            "store": "IKEA",
            "mock_url": "https://example.com/ikea/linnmon_desk",
        },
        {
            "name": "Wood Computer Desk",
            "store": "Walmart",
            "mock_url": "https://example.com/walmart/wood_desk",
        },
        {"name": "Generic Desk", "store": "Generic", "mock_url": "N/A"},
    ],
    "chair": [
        {
            "name": "Adde Chair",
            "store": "IKEA",
            "mock_url": "https://example.com/ikea/adde_chair",
        },
        {
            "name": "Accent Chair",
            "store": "Home Depot",
            "mock_url": "https://example.com/homedepot/accent_chair",
        },
        {"name": "Generic Chair", "store": "Generic", "mock_url": "N/A"},
    ],
    "lamp": [
        {
            "name": "Lersta Floor Lamp",
            "store": "IKEA",
            "mock_url": "https://example.com/ikea/lersta_lamp",
        },
        {
            "name": "Desk Lamp",
            "store": "Target",
            "mock_url": "https://example.com/target/desk_lamp",
        },
        {"name": "Generic Lamp", "store": "Generic", "mock_url": "N/A"},
    ],
}


# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8 object detection model."""
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")  # yolov8n is the nano model, good for local/CPU
        st.success("YOLOv8 model loaded successfully.")
        return model
    except Exception as e:
        st.error(
            f"Failed to load YOLOv8 model: {e}. Please ensure `ultralytics` is installed and a stable internet connection for initial download."
        )
        st.stop()


@st.cache_resource
def load_diffusion_models():
    """Loads Stable Diffusion and ControlNet models."""
    try:
        from diffusers import (
            StableDiffusionControlNetPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )
        from transformers import pipeline
        import torch

        # Load ControlNet model (trained on segmentation maps)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16
        )

        # Load Stable Diffusion pipeline with ControlNet
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Optimize for Apple Silicon (MPS) if available
        if torch.backends.mps.is_available():
            pipe.to("mps")
            st.success(
                "Stable Diffusion and ControlNet loaded successfully on MPS (Apple Silicon)."
            )
        else:
            pipe.to("cpu")  # Fallback to CPU
            st.warning(
                "MPS (Apple Silicon GPU) not available. Models loading on CPU, which will be significantly slower."
            )

        return pipe
    except Exception as e:
        st.error(
            f"Failed to load Stable Diffusion/ControlNet models: {e}. Please ensure `diffusers`, `transformers`, `torch` are installed and a stable internet connection for initial download."
        )
        st.stop()


# --- AI Logic Functions ---


def analyze_room_and_detect_objects(image_files_dict):
    """
    Stage 1: Room Analysis and Scene Understanding (Simplified)
    Identifies existing large objects in the uploaded images using YOLOv8.
    Scene understanding (room_type, colors, lighting) is left to be inferred by LLM
    or explicitly provided by the user due to complexity constraints.
    """
    yolo_model = load_yolo_model()
    room_analysis_data = {
        "existing_objects": [],
        "inferred_style_cues": [],
        "wall_images": {},  # This will store PIL Image objects, keyed by the standardized wall_id
    }

    st.subheader("🖼️ Analyzing Room Images...")
    for (
        wall_id,
        img_file_obj,
    ) in image_files_dict.items():  # img_file_obj is a Streamlit UploadedFile object
        # Open the uploaded file object as a PIL Image
        image = Image.open(img_file_obj).convert("RGB")
        room_analysis_data["wall_images"][wall_id] = image  # Store original PIL Image

        # Save image to a BytesIO object for YOLO
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Perform object detection
        results = yolo_model(image, stream=True)  # stream=True for generator output

        wall_objects = []
        for r in results:
            if r.boxes:  # Check if any boxes were detected
                for *xyxy, conf, cls in r.boxes.data:
                    label = yolo_model.names[int(cls)]  # Get class name
                    x1, y1, x2, y2 = map(int, xyxy)
                    width = x2 - x1
                    height = y2 - y1
                    approx_size = "small"
                    if width * height > 0.3 * image.width * image.height:
                        approx_size = "large"
                    elif width * height > 0.1 * image.width * image.height:
                        approx_size = "medium"

                    wall_objects.append(
                        {
                            "type": label,
                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2,
                            ],  # Storing bounding box for potential future use
                            "approx_location_wall": wall_id,
                            "approx_size": approx_size,
                        }
                    )
        room_analysis_data["existing_objects"].extend(wall_objects)
        st.write(f"Detected {len(wall_objects)} objects on {wall_id}.")

    # Deduplicate existing objects if multiple walls show the same large object
    # This is a heuristic and might need refinement for complex layouts
    deduplicated_objects = {}
    for obj in room_analysis_data["existing_objects"]:
        # Use a combination of type and a simplified location (e.g., wall_id) for deduplication
        # This prevents counting the same physical object seen on different walls multiple times.
        # A more robust solution might involve 3D reconstruction or more advanced matching.
        key = f"{obj['type']}_{obj['approx_location_wall']}_{obj['approx_size']}"
        if key not in deduplicated_objects:
            deduplicated_objects[key] = obj
    room_analysis_data["existing_objects"] = list(deduplicated_objects.values())

    return room_analysis_data


def generate_layout_prompt(room_analysis_data, user_prefs):
    """
    Stage 2: User Preference Integration & Prompt Generation
    Combines room analysis with user preferences to create a prompt for the LLM.
    """
    existing_obj_str = ", ".join(
        [
            f"{obj['type']} ({obj['approx_size']})"
            for obj in room_analysis_data["existing_objects"]
        ]
    )
    if not existing_obj_str:
        existing_obj_str = "no large physical objects"

    store_prefs_str = (
        ", ".join(user_prefs["store_preferences"])
        if user_prefs["store_preferences"]
        else "generic alternatives"
    )

    # Get the actual wall IDs from the uploaded images (now guaranteed to be 'wall_a', 'wall_b', etc.)
    available_wall_ids = list(room_analysis_data["wall_images"].keys())
    walls_info = ", ".join(available_wall_ids)

    # Prepare a simple example JSON snippet that uses one of the actual wall IDs
    example_wall_id = available_wall_ids[0] if available_wall_ids else "wall_a"
    example_json_snippet = f"""
    ```json
    {{
      "proposed_layout_description": "A Scandinavian-style bedroom with a centralized bed and cozy lighting.",
      "wall_configs": [
        {{
          "wall_id": "{example_wall_id}",
          "items": [
            {{"type": "bed", "style": "minimalist", "approx_position": "center", "product_suggestion": {{"name": "Malm Bed Frame", "store": "IKEA", "mock_url": "[https://example.com/ikea/malm](https://example.com/ikea/malm)"}}}},
            {{"type": "nightstand", "style": "circular", "approx_position": "left_of_bed", "product_suggestion": {{"name": "Generic Round Table", "store": "Generic", "mock_url": "N/A"}}}}
          ]
        }}
        // Add more wall configurations for other walls if applicable, e.g., for "wall_b"
      ]
    }}
    ```
    """

    prompt = f"""
    Design a {user_prefs['room_type']} with a {user_prefs['desired_style']} aesthetic,
    using {user_prefs['color_palette']} and focusing on {user_prefs['material_preferences']}.
    The current room has existing large objects: {existing_obj_str}.

    Considering the renovation level: "{user_prefs['renovation_level']}".
    - If "Rearrange Existing Objects Only", focus on moving detected objects.
    - If "Minor Furniture Replacement/Addition", replace some existing items and add new ones.
    - If "Major Overhaul/Construction", you have freedom to remove/add structural elements (e.g., paint walls, add built-in shelves) and new furniture.

    {user_prefs['custom_prompt']}

    Suggest items primarily from {store_prefs_str}.
    **Crucially, provide the layout ONLY for the following walls: {walls_info}.**
    For each of these walls, provide its configuration. Ensure the `wall_id` matches exactly one of these: {walls_info}.

    Provide the layout as a JSON object with a `proposed_layout_description` and a `wall_configs` array.
    Each `wall_config` should have a `wall_id` (matching one from {walls_info}) and an `items` array.
    Each item should have `type`, `style`, `approx_position` (e.g., "center", "left_side", "right_corner", "above_bed", "along_wall"), and `product_suggestion` (with `name`, `store`, `mock_url`).
    Ensure `approx_position` is descriptive enough to guide image generation for approximate placement.

    Example JSON structure (note: `wall_id`s in the example are illustrative, use the actual provided wall IDs):
    {example_json_snippet}
    """
    return prompt


def generate_layout_and_products(prompt, iteration_count=0):
    """
    Stage 3: Layout Generation & Object Placement (LLM + Constraint Solver)
    Uses Gemini-Pro to generate the layout based on the prompt.
    Includes a very basic product linking mechanism.
    """
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        st.error(
            "Please provide your Gemini API key in the `GEMINI_API_KEY` variable in the script."
        )
        st.stop()

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {
        "contents": chat_history,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "proposed_layout_description": {"type": "STRING"},
                    "wall_configs": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "wall_id": {"type": "STRING"},
                                "items": {
                                    "type": "ARRAY",
                                    "items": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "type": {"type": "STRING"},
                                            "style": {"type": "STRING"},
                                            "approx_position": {"type": "STRING"},
                                            "product_suggestion": {
                                                "type": "OBJECT",
                                                "properties": {
                                                    "name": {"type": "STRING"},
                                                    "store": {"type": "STRING"},
                                                    "mock_url": {"type": "STRING"},
                                                },
                                                "propertyOrdering": [
                                                    "name",
                                                    "store",
                                                    "mock_url",
                                                ],
                                            },
                                        },
                                        "propertyOrdering": [
                                            "type",
                                            "style",
                                            "approx_position",
                                            "product_suggestion",
                                        ],
                                    },
                                },
                            },
                            "propertyOrdering": ["wall_id", "items"],
                        },
                    },
                },
                "propertyOrdering": ["proposed_layout_description", "wall_configs"],
            },
        },
    }

    # Add a variation instruction for 'more suggestions'
    if iteration_count > 0:
        payload["contents"][0]["parts"][0][
            "text"
        ] += f"\n\nGenerate a different layout suggestion. This is attempt #{iteration_count + 1}."

    with st.spinner("Generating layout suggestions (this may take a moment)..."):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            result = response.json()

            if (
                result.get("candidates")
                and result["candidates"][0].get("content")
                and result["candidates"][0]["content"].get("parts")
            ):
                json_string = result["candidates"][0]["content"]["parts"][0]["text"]
                # Sometimes the LLM might include markdown fences even with schema
                if json_string.startswith("```json") and json_string.endswith("```"):
                    json_string = json_string[7:-3].strip()
                parsed_json = json.loads(json_string)

                # --- Basic Product Linking based on mock DB ---
                for wall_config in parsed_json["wall_configs"]:
                    for item in wall_config["items"]:
                        item_type = item["type"].lower().replace(" ", "_")
                        linked_product = None
                        if item_type in MOCK_PRODUCT_DATABASE:
                            # Try to match a store if specified in item["product_suggestion"] from LLM
                            # Otherwise, pick a random one or generic
                            if item.get("product_suggestion") and item[
                                "product_suggestion"
                            ].get("store"):
                                for prod in MOCK_PRODUCT_DATABASE[item_type]:
                                    if (
                                        prod["store"].lower()
                                        == item["product_suggestion"]["store"].lower()
                                    ):
                                        linked_product = prod
                                        break
                            if (
                                linked_product is None
                            ):  # If no store match or no specific store in LLM output
                                # Pick the first available product or generic
                                for prod in MOCK_PRODUCT_DATABASE[item_type]:
                                    if prod["store"] == "Generic":
                                        linked_product = prod
                                        break
                                if (
                                    linked_product is None
                                    and MOCK_PRODUCT_DATABASE[item_type]
                                ):
                                    linked_product = MOCK_PRODUCT_DATABASE[item_type][
                                        0
                                    ]  # Fallback to first available

                            if linked_product:
                                item["product_suggestion"] = linked_product
                            else:  # Fallback to a generic item if no match at all
                                item["product_suggestion"] = {
                                    "name": f"Generic {item['type']}",
                                    "store": "Generic",
                                    "mock_url": "N/A",
                                }
                        else:  # If item type not in our mock database
                            item["product_suggestion"] = {
                                "name": f"Generic {item['type']}",
                                "store": "Generic",
                                "mock_url": "N/A",
                            }

                return parsed_json
            else:
                st.error(
                    "Gemini API returned an unexpected response structure or no content."
                )
                st.json(result)
                return None
        except requests.exceptions.RequestException as e:
            st.error(
                f"Error calling Gemini API: {e}. Please check your API key and network connection."
            )
            return None
        except json.JSONDecodeError as e:
            st.error(
                f"Error parsing LLM response as JSON: {e}. Raw response: {json_string}"
            )
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during layout generation: {e}")
            return None


def generate_images_with_controlnet(original_wall_images, proposed_layout):
    """
    Stage 4: Image Generation (Text-to-Image with ControlNet)
    Generates new images based on the proposed layout and original images.
    Uses a simplified ControlNet conditioning (drawing rectangles on a blank map)
    for new objects.
    """
    pipe = load_diffusion_models()
    generated_images_data = {}

    st.subheader("🎨 Generating AI Design Images...")
    for wall_config in proposed_layout["wall_configs"]:
        wall_id = wall_config[
            "wall_id"
        ]  # This wall_id now correctly matches 'wall_a', 'wall_b' etc.
        if wall_id not in original_wall_images:
            st.warning(
                f"Original image for {wall_id} not found based on LLM's suggested wall_id. Skipping image generation for this wall. Please ensure LLM output matches uploaded walls."
            )
            continue  # Skip if the LLM hallucinated a wall_id not provided by the user

        original_image = original_wall_images[
            wall_id
        ].copy()  # Get the PIL Image object
        width, height = original_image.size

        # Create a blank control image (segmentation map proxy)
        control_image = Image.new(
            "RGB", (width, height), color=(0, 0, 0)
        )  # Black background
        draw = ImageDraw.Draw(control_image)
        # Use a simple font for drawing on control image (if text needed)
        try:
            font = ImageFont.truetype(
                "arial.ttf", 20
            )  # This might fail if font not found
        except IOError:
            font = ImageFont.load_default()  # Fallback to default

        # Draw proposed items as simple rectangles/shapes on the control_image
        # This is a simplification of true semantic segmentation for ControlNet
        # The quality of output highly depends on how well ControlNet interprets these simple shapes.
        object_colors = {
            "bed": (255, 0, 0),
            "sofa": (0, 255, 0),
            "nightstand": (0, 0, 255),
            "art_piece": (255, 255, 0),
            "desk": (0, 255, 255),
            "chair": (255, 0, 255),
            "lamp": (255, 128, 0),
            "rug": (128, 255, 0),
            "plant": (0, 128, 255),
            "bookshelf": (128, 0, 255),
            "table": (255, 0, 128),  # Add more colors for other object types
        }

        for item in wall_config["items"]:
            item_type = item["type"].lower().replace(" ", "_")
            pos = item["approx_position"].lower()
            color = object_colors.get(item_type, (128, 128, 128))  # Default grey

            # Very basic approximation of object placement based on text position
            # This is heuristic and will require tuning/more sophisticated logic for real use
            x_center, y_center = width // 2, height // 2
            item_width, item_height = (
                width // 4,
                height // 4,
            )  # Default size for new objects

            # More refined placement logic based on common interior design item positions
            if "bed" in item_type:
                item_width, item_height = int(width * 0.7), int(height * 0.4)
                x1, y1 = (
                    width - item_width
                ) // 2, height - item_height - 20  # bottom center
            elif "sofa" in item_type:
                item_width, item_height = int(width * 0.7), int(height * 0.3)
                x1, y1 = (
                    width - item_width
                ) // 2, height - item_height - 20  # bottom center
            elif "nightstand" in item_type:
                item_width, item_height = int(width * 0.15), int(height * 0.25)
                if "left_of_bed" in pos or "left" in pos:
                    x1 = int(width * 0.1)  # Arbitrary left side
                elif "right_of_bed" in pos or "right" in pos:
                    x1 = int(width * 0.75)  # Arbitrary right side
                else:  # Default to one side if not specified
                    x1 = int(width * 0.1)
                y1 = height - item_height - 20
            elif "art_piece" in item_type:
                item_width, item_height = int(width * 0.4), int(height * 0.3)
                if "above_sofa" in pos or "above_bed" in pos:
                    x1 = (width - item_width) // 2
                    y1 = int(height * 0.2)  # Higher up
                else:
                    x1 = (width - item_width) // 2
                    y1 = int(height * 0.3)
            elif "desk" in item_type:
                item_width, item_height = int(width * 0.5), int(height * 0.35)
                x1, y1 = (width - item_width) // 2, height - item_height - 20
            elif "chair" in item_type:
                item_width, item_height = int(width * 0.25), int(height * 0.4)
                if "corner" in pos:
                    x1 = int(width * 0.1) if "left" in pos else int(width * 0.65)
                else:
                    x1 = (width - item_width) // 2
                y1 = height - item_height - 20
            elif "lamp" in item_type:
                item_width, item_height = int(width * 0.1), int(height * 0.5)
                if "corner" in pos or "side" in pos:
                    x1 = int(width * 0.05) if "left" in pos else int(width * 0.85)
                else:  # Default to one side
                    x1 = int(width * 0.05)
                y1 = height - item_height - 20
            elif "rug" in item_type:
                item_width, item_height = int(width * 0.8), int(height * 0.6)
                x1, y1 = (
                    width - item_width
                ) // 2, height - item_height + 50  # on the floor
            elif "plant" in item_type:
                item_width, item_height = int(width * 0.1), int(height * 0.3)
                if "corner" in pos:
                    x1 = int(width * 0.05) if "left" in pos else int(width * 0.85)
                else:
                    x1 = x_center - item_width // 2
                y1 = height - item_height - 20
            elif "bookshelf" in item_type:
                item_width, item_height = int(width * 0.4), int(height * 0.6)
                x1 = (
                    int(width * 0.1) if "left" in pos else int(width * 0.5)
                )  # Example for side placement
                y1 = height - item_height - 20
            elif "table" in item_type:
                item_width, item_height = int(width * 0.4), int(height * 0.3)
                x1, y1 = (width - item_width) // 2, height - item_height - 20
            else:  # Generic object placement
                item_width, item_height = int(width * 0.3), int(height * 0.3)
                x1, y1 = (width - item_width) // 2, (
                    height - item_height
                ) // 2  # Center

            x2, y2 = x1 + item_width, y1 + item_height
            draw.rectangle([x1, y1, x2, y2], fill=color)

        # Main prompt for ControlNet
        prompt_text = f"Photorealistic interior design, {proposed_layout['proposed_layout_description']}, {user_prefs['desired_style']} style, {user_prefs['color_palette']} color palette, {user_prefs['material_preferences']} materials, high detail."
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, noisy"

        with st.spinner(f"Generating image for {wall_id} using ControlNet..."):
            try:
                # Generate image
                generated_image = pipe(
                    prompt_text,
                    image=control_image,  # ControlNet uses this as conditioning
                    num_inference_steps=20,  # Reduced steps for faster generation, can increase for quality
                    # guidance_scale=7.5, # Often a good default
                    # generator=torch.Generator("cpu").manual_seed(0) # For reproducibility
                ).images[0]

                generated_images_data[wall_id] = {
                    "image": generated_image,
                    "items": wall_config["items"],
                }
            except Exception as e:
                st.error(f"Error generating image for {wall_id} with ControlNet: {e}")
                # Fallback to original image if generation fails
                generated_images_data[wall_id] = {
                    "image": original_image,
                    "items": wall_config["items"],
                }
    return generated_images_data


def overlay_labels_and_display(generated_images_data):
    """
    Stage 5: Product Linking and Labeling Overlay
    Overlays labels on generated images and displays product info.
    """
    st.subheader("✨ Your AI-Generated Interior Design Suggestions!")

    for wall_id, data in generated_images_data.items():
        image = data["image"].copy()
        items = data["items"]

        draw = ImageDraw.Draw(image)
        try:
            # Adjust font size for better visibility on images
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()

        # Simple logic to place labels. This will need more refinement for robust placement.
        # For this simplified implementation, we don't have exact pixel coordinates of new objects.
        # We'll just place labels based on general areas or infer from item type/position.
        label_offset_y = 30
        for i, item in enumerate(items):
            label_text = f"{item['type']}"
            if item.get("product_suggestion"):
                label_text += f"\n- {item['product_suggestion']['name']} ({item['product_suggestion']['store']})"

            # Placeholder for label position. In a real app, this would be based on generated object bbox.
            # Here, we'll just stack them or place them in a corner.
            text_x = 20
            text_y = 20 + i * label_offset_y * 2  # Place labels vertically

            # Add a background rectangle for better readability
            # textbbox calculates the bounding box of the text
            try:
                text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
                padding = 5
                draw.rectangle(
                    (
                        text_bbox[0] - padding,
                        text_bbox[1] - padding,
                        text_bbox[2] + padding,
                        text_bbox[3] + padding,
                    ),
                    fill=(0, 0, 0, 150),  # Semi-transparent black
                )
                draw.text((text_x, text_y), label_text, font=font, fill=(255, 255, 255))
            except Exception as e:
                st.warning(f"Could not draw text label for item '{item['type']}': {e}")

        st.image(
            image, caption=f"Suggested Design for {wall_id}", use_column_width=True
        )
        st.markdown(f"**Items for {wall_id}:**")
        for item in items:
            product_info = item.get("product_suggestion", {})
            st.markdown(
                f"- **{item['type']}** ({item['style']}): {product_info.get('name', 'N/A')} from {product_info.get('store', 'N/A')}"
            )
            if product_info.get("mock_url") and product_info["mock_url"] != "N/A":
                st.markdown(f"  [View Product (Mock)]({product_info['mock_url']})")
        st.markdown("---")


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Interior Designer")

st.title("🏡 AI Interior Design Assistant")
st.markdown(
    "Upload photos of your room's walls, tell me your preferences, and I'll generate design suggestions!"
)

# --- 1. User Inputs: Upload Images ---
st.header("1. Upload Your Room Wall Photos")
uploaded_files = st.file_uploader(
    "Upload photos of each wall (e.g., Wall A, Wall B, etc.). Max 4 images recommended for performance.",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# Use a dictionary to store uploaded file objects with standardized wall IDs
if "uploaded_file_objects" not in st.session_state:
    st.session_state.uploaded_file_objects = {}

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        # FIX: Standardize wall_id to lowercase for consistency with LLM's common output patterns
        wall_id = f"wall_{chr(97 + i)}"  # Generates "wall_a", "wall_b", etc.
        st.session_state.uploaded_file_objects[wall_id] = uploaded_file
        st.image(uploaded_file, caption=f"Uploaded: {wall_id}", width=200)

# --- 2. User Inputs: Preferences ---
st.header("2. Tell Me Your Design Preferences")

user_prefs = {}

user_prefs["room_type"] = st.selectbox(
    "What type of room is this?",
    (
        "Living Room",
        "Bedroom",
        "Kitchen",
        "Dining Room",
        "Home Office",
        "Bathroom",
        "Other",
    ),
)

user_prefs["renovation_level"] = st.selectbox(
    "What's your renovation level?",
    (
        "Rearrange Existing Objects Only",
        "Minor Furniture Replacement/Addition",
        "Major Overhaul/Construction",
    ),
)

user_prefs["desired_style"] = st.selectbox(
    "What's your desired interior style?",
    (
        "Modern",
        "Minimalist",
        "Scandinavian",
        "Bohemian",
        "Industrial",
        "Classic",
        "Farmhouse",
        "Coastal",
        "Eclectic",
        "Traditional",
    ),
)

user_prefs["color_palette"] = st.text_input(
    "Preferred Color Palette (e.g., 'Earthy Tones', 'Cool Blues', 'Vibrant Accents')",
    value="neutral colors with warm accents",
)

user_prefs["material_preferences"] = st.text_input(
    "Material Preferences (e.g., 'Wood', 'Metal', 'Fabric', 'Glass')",
    value="natural wood and soft fabrics",
)

st.subheader("Where do you prefer to shop for products?")
all_stores = [
    "IKEA",
    "Amazon",
    "Target",
    "Wayfair",
    "Walmart",
    "Home Depot",
    "Generic Items Only",
]
user_prefs["store_preferences"] = st.multiselect(
    "Select preferred stores (or 'Generic Items Only'):",
    options=all_stores,
    default=["Generic Items Only"],
)

user_prefs["custom_prompt"] = st.text_area(
    "Any additional custom requests or ideas?",
    value="Add more plants and create a cozy reading nook.",
)

st.session_state.user_prefs = user_prefs  # Store preferences in session state

# --- Main Logic Trigger ---
if st.button("Generate Design Suggestions", type="primary"):
    if not st.session_state.get(
        "uploaded_file_objects"
    ):  # Check if any files were uploaded and stored
        st.error("Please upload at least one room wall photo to get started.")
    else:
        st.session_state.current_iteration = 0
        with st.spinner("Starting design generation..."):
            # Stage 1: Room Analysis (Pass the dictionary of UploadedFile objects)
            st.session_state.room_analysis_data = analyze_room_and_detect_objects(
                st.session_state.uploaded_file_objects
            )

            if st.session_state.room_analysis_data:
                # Stage 2: Prompt Generation
                llm_prompt = generate_layout_prompt(
                    st.session_state.room_analysis_data, st.session_state.user_prefs
                )

                # Stage 3: Layout Generation (LLM)
                st.session_state.current_layout_data = generate_layout_and_products(
                    llm_prompt
                )

                if st.session_state.current_layout_data:
                    # Stage 4 & 5: Image Generation and Labeling
                    # Now, original_wall_images (room_analysis_data["wall_images"]) holds PIL Image objects
                    st.session_state.generated_images_data = (
                        generate_images_with_controlnet(
                            st.session_state.room_analysis_data["wall_images"],
                            st.session_state.current_layout_data,
                        )
                    )
                    overlay_labels_and_display(st.session_state.generated_images_data)
                else:
                    st.error(
                        "Failed to generate a valid layout from AI. Please try again or refine your prompt."
                    )
            else:
                st.error(
                    "Failed to analyze room images. Please ensure valid image files were uploaded."
                )

# --- "More Suggestions" Button ---
if st.session_state.get("current_layout_data") and st.session_state.get(
    "room_analysis_data"
):
    if st.button("Generate More Suggestions"):
        st.session_state.current_iteration = (
            st.session_state.get("current_iteration", 0) + 1
        )
        with st.spinner(
            f"Generating alternative layout suggestion (Attempt #{st.session_state.current_iteration + 1})..."
        ):
            llm_prompt = generate_layout_prompt(
                st.session_state.room_analysis_data, st.session_state.user_prefs
            )
            st.session_state.current_layout_data = generate_layout_and_products(
                llm_prompt, iteration_count=st.session_state.current_iteration
            )
            if st.session_state.current_layout_data:
                st.session_state.generated_images_data = (
                    generate_images_with_controlnet(
                        st.session_state.room_analysis_data["wall_images"],
                        st.session_state.current_layout_data,
                    )
                )
                overlay_labels_and_display(st.session_state.generated_images_data)
            else:
                st.error("Failed to generate more suggestions. Please try again.")
