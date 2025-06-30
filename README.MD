# **Functional Specification: AI-Powered Interior Design Suggestion Generation Logic**

## **1\. Introduction**

This document outlines the functional specification for the core AI-driven interior design suggestion generation logic within the mobile application. The aim is to transform user-provided room wall photos and preferences into AI-generated visual design suggestions, complete with product labeling.

## **2\. Overall Architecture Overview**

The system employs a multi-stage pipeline, leveraging a combination of computer vision models for scene understanding, a large language model (LLM) for creative layout generation, and a text-to-image diffusion model with conditioning for visual rendering.

## **3\. Core Algorithm: Interior Design Suggestion Generation**

### **3.1. Stage 1: Room Analysis and Scene Understanding**

* **Purpose:** To extract comprehensive information about the user's existing room, including its current state, existing objects, and inherent characteristics, from provided photographs.  
* **Input:**  
  * Multiple JPEG/PNG image files of each wall of the room (e.g., "wall\_A.jpg", "wall\_B.jpg", "wall\_C.jpg", "wall\_D.jpg"). Each photo is taken from the center of the room.  
  * Metadata (e.g., photo orientation if available, though not a prime concern).  
* **Processing:**  
  1. **Image Feature Extraction:** Each input image is processed through a Vision Transformer (ViT) or similar pre-trained encoder (e.g., from HuggingFace transformers library) to generate high-level feature representations.  
  2. **Object Detection & Localization:** A pre-trained object detection model (e.g., YOLOv8, available on HuggingFace Hub and runnable via ultralytics library) is applied to each image.  
     * **Task:** Identify and localize large physical objects (e.g., bed, sofa, desk, cabinet, window, door, fireplace) present in the room.  
     * **Output:** Bounding box coordinates and class labels for each detected object, along with confidence scores.  
  3. **Scene Understanding & Contextualization:**  
     * An image captioning or visual question answering (VQA) model (e.g., BLIP-2 or ViLBERT, available on HuggingFace) or a smaller, fine-tuned model for room types/styles is used to infer the room's type (e.g., "bedroom", "living room", "kitchen"), dominant color schemes, lighting conditions, and potential existing design styles.  
     * **Note:** While not strictly 3D, relative spatial relationships between detected objects (e.g., "bed is against the wall") and approximate wall dimensions can be inferred from multiple 2D views and object sizes.  
* **Output:**  
  * A structured JSON or textual representation summarizing the analyzed room:  
    {  
      "room\_type": "bedroom",  
      "dominant\_colors": \["beige", "white", "brown"\],  
      "lighting\_condition": "bright, natural light",  
      "existing\_objects": \[  
        {"id": "obj\_001", "type": "bed", "approx\_location\_wall": "wall\_A", "approx\_size": "large"},  
        {"id": "obj\_002", "type": "nightstand", "approx\_location\_wall": "wall\_A", "approx\_size": "small"},  
        {"id": "obj\_003", "type": "window", "approx\_location\_wall": "wall\_B", "approx\_size": "medium"}  
      \],  
      "inferred\_style\_cues": \["modern", "minimalist"\]  
    }

  * Intermediate outputs like segmented masks or bounding box overlays (for internal model use, not directly exposed to user).

### **3.2. Stage 2: User Preference Integration & Prompt Generation**

* **Purpose:** To synthesize the analyzed room data with user-defined preferences into a comprehensive prompt for the layout generation and image generation stages.  
* **Input:**  
  * Output from Stage 1 (structured room analysis data).  
  * User inputs (via UI elements):  
    * Renovation Level: (e.g., "Rearrange Existing Objects Only", "Minor Furniture Replacement/Addition", "Major Overhaul/Construction").  
    * Desired Style: (e.g., "Bohemian", "Scandinavian", "Industrial", "Classic").  
    * Preferred Color Palette: (e.g., "Earthy Tones", "Cool Blues", "Vibrant Accents").  
    * Material Preferences: (e.g., "Wood", "Metal", "Fabric", "Glass").  
    * Store Preferences (checklist): (e.g., IKEA, Amazon, Target, Wayfair, Walmart, Home Depot, "Generic Items Only").  
    * Optional Custom Text Prompt: (e.g., "add a reading nook", "make it cozy").  
* **Processing:**  
  1. **Conditional Logic for Renovation Level:**  
     * If "Rearrange Existing Objects Only": The system prioritizes using identified existing objects and suggests new layouts for them.  
     * If "Minor Furniture Replacement/Addition": Allows for replacing existing items and introducing new ones.  
     * If "Major Overhaul/Construction": Gives the AI more freedom to suggest significant changes, potentially removing existing items and proposing new structural elements (e.g., new wall colors, built-in shelves).  
  2. **Prompt Engineering:**  
     * The structured room analysis and user preferences are combined into a detailed natural language prompt for the LLM.  
     * Example Prompt Structure: "Design a \[room\_type\] with a \[desired\_style\] aesthetic, using \[preferred\_color\_palette\] and \[material\_preferences\]. The current room has \[list\_existing\_objects\_with\_approx\_location\]. \[Custom\_text\_prompt\_if\_any\]. Consider the renovation level: \[renovation\_level\]. Suggest items primarily from \[store\_preferences\] or generic alternatives."  
* **Output:** A highly detailed and contextualized text prompt, ready for the LLM.

### **3.3. Stage 3: Layout Generation & Object Placement (LLM \+ Constraint Solver)**

* **Purpose:** To generate a textual description of a new interior layout, including proposed new items and their approximate placement, while considering existing objects and user constraints.  
* **Input:**  
  * Contextualized text prompt from Stage 2\.  
  * Structured room analysis data from Stage 1 (especially existing objects).  
* **Processing:**  
  1. **Initial Layout Concept (LLM):**  
     * An LLM (e.g., Gemini-Pro, which is free to use via API with an API key, or a smaller fine-tuned model if available) receives the engineered prompt.  
     * **Task:** The LLM generates a textual description of a new layout, proposing furniture types, decorative elements, and their conceptual placement within the room. It considers the renovation\_level to decide whether to suggest moving, removing, or adding objects. It also attempts to incorporate store\_preferences by suggesting item types commonly found in those stores, or generic descriptions if specific stores aren't preferred or a direct match isn't found.  
     * **Example LLM Output Segment:** "For Wall A, suggest a queen-sized bed with a minimalist frame, centered. To its left, a small circular nightstand. For Wall B, a comfortable two-seater sofa with an abstract art piece above it. Ensure enough space for walking."  
  2. **Object Instantiation & Refinement (LLM \+ Constraint Solver/Rule-based System):**  
     * The LLM's conceptual layout is further refined. For items, it may suggest specific types or general descriptions (e.g., "modern desk lamp," "velvet armchair").  
     * A rule-based system or a simple constraint solver (implemented in Python) uses the inferred room dimensions and existing object locations to translate the LLM's conceptual placements into more precise, approximate coordinates or relative positions suitable for image generation.  
     * **Constraint Checking:** This module ensures basic spatial feasibility (e.g., objects don't overlap excessively, pathways are clear, objects are placed against walls unless specified).  
     * **Iterative Refinement (Optional but beneficial):** The LLM could be prompted iteratively if initial layouts are deemed spatially problematic by the constraint solver, allowing for adjustment.  
     * **Product Linking (Preliminary):** Based on store\_preferences and suggested item types, a lookup mechanism (e.g., pre-indexed data or a simple search function against a mock database of product categories from selected stores) attempts to associate proposed items with *example* generic product names or categories, and *mock* URLs.  
* **Output:**  
  * A structured JSON data block describing the proposed new layout, ready for image generation:  
    {  
      "proposed\_layout\_description": "A Scandinavian-style bedroom with a centralized bed and cozy lighting.",  
      "wall\_configs": \[  
        {  
          "wall\_id": "wall\_A",  
          "items": \[  
            {"type": "bed", "style": "minimalist", "approx\_position": "center", "product\_suggestion": {"name": "Malm Bed Frame", "store": "IKEA", "mock\_url": "https://example.com/ikea/malm"}},  
            {"type": "nightstand", "style": "circular", "approx\_position": "left\_of\_bed", "product\_suggestion": {"name": "Generic Round Table", "store": "Generic", "mock\_url": "N/A"}}  
          \]  
        },  
        {  
          "wall\_id": "wall\_B",  
          "items": \[  
            {"type": "sofa", "style": "two-seater", "approx\_position": "center", "product\_suggestion": {"name": "Ektorp Sofa", "store": "IKEA", "mock\_url": "https://example.com/ikea/ektorp"}},  
            {"type": "art\_piece", "style": "abstract", "approx\_position": "above\_sofa", "product\_suggestion": {"name": "Abstract Canvas Print", "store": "Amazon", "mock\_url": "https://example.com/amazon/art"}}  
          \]  
        }  
      \]  
    }

  * If the user requests "more" suggestions, the LLM is re-prompted with an instruction to generate an alternative layout.

### **3.4. Stage 4: Image Generation (Text-to-Image with ControlNet)**

* **Purpose:** To visually render the proposed interior design configuration onto the original room images, leveraging their existing structure.  
* **Input:**  
  * Original room wall images from Stage 1\.  
  * Structured proposed layout data from Stage 3\.  
* **Processing:**  
  1. **Control Map Generation:** For each wall:  
     * Based on the proposed\_layout data, generate ControlNet-compatible conditioning images. This typically involves:  
       * **Segmentation Map (Semantic Segmentation):** Create a blank image and "draw" masks for each proposed object type (e.g., bed, sofa, table) at their approximate positions, using distinct colors for different object classes. This can be done programmatically using libraries like PIL/OpenCV.  
       * **Optionally, Depth Map:** If possible to infer crude depth information from bounding box sizes and relative positions from 2D images, a simple depth map could be generated (though this is more complex and might be omitted for simplicity given the constraints).  
     * **Integration of Existing Objects:** The generated control maps *must* incorporate the detected existing\_objects from Stage 1, either by keeping them as-is (if renovation\_level is "Rearrange") or by marking them for removal/replacement based on the new layout.  
  2. **Text-to-Image Generation (ControlNet):**  
     * A pre-trained text-to-image diffusion model (e.g., Stable Diffusion, specifically with a ControlNet module loaded for conditional image generation, runnable via HuggingFace diffusers library) is used.  
     * **Inputs to ControlNet:**  
       * The original image of the wall (as an init\_image for in-painting or image-to-image).  
       * The generated ControlNet conditioning image (e.g., segmentation map).  
       * A detailed text prompt combining the proposed\_layout\_description with stylistic elements and the desire for photorealism (e.g., "a \[room\_type\] in \[desired\_style\], featuring \[list\_proposed\_items\], photorealistic, high detail").  
     * **Task:** The model generates a new image of the wall, incorporating the proposed layout while maintaining the overall structure, lighting, and non-modified background elements from the original image, guided by the control map.  
* **Output:** AI-generated images of the suggested room configuration for each wall (e.g., "suggestion\_wall\_A\_1.jpg", "suggestion\_wall\_B\_1.jpg").

### **3.5. Stage 5: Product Linking and Labeling Overlay**

* **Purpose:** To visually annotate the generated design suggestions with labels for proposed items and link them to product information.  
* **Input:**  
  * AI-generated images from Stage 4\.  
  * Structured proposed layout data from Stage 3 (containing proposed item types, names, store info, and mock URLs).  
* **Processing:**  
  1. **Label Position Estimation:** Based on the approximate positions defined in the proposed\_layout data, overlay text labels onto the generated images at appropriate locations near the corresponding items.  
  2. **Product Information Overlay:**  
     * For each labeled item, display its type/name (e.g., "Bed", "Two-seater Sofa").  
     * If a specific product suggestion was found (not generic), display the store name and potentially a simplified product name.  
     * The full mock URL is stored but not necessarily displayed directly on the image itself, rather accessible on tap or via a linked product list.  
* **Output:**  
  * Final AI-generated images with overlaid text labels for each proposed item.  
  * Associated metadata for each image, linking labels to the detailed product suggestions (name, store, mock URL).

## **4\. Model Constraints and Considerations**

* **Model Selection:** Emphasis on models available on HuggingFace Hub that can be run with free tiers/APIs (e.g., Gemini-Pro API key to be provided, open-source models like Stable Diffusion, YOLOv8).  
* **Hardware Limitations:** The architecture considers the m3 Mac laptop constraint, favoring models that are relatively efficient or can be run on smaller VRAM (e.g., diffusers library allows for CPU fallback or smaller model variants, though this will impact generation speed). Local inference is preferred where possible to avoid API costs beyond LLM.  
* **Scalability (Local Scope):** This functional spec is designed for a moderately small, local scope college project, not a highly scalable commercial application.  
* **"More Suggestions" Feature:** When the user requests "more," the LLM in Stage 3 will be re-queried with a variation prompt to generate an alternative layout, and the entire pipeline (Stages 3, 4, 5\) will be re-executed for the new suggestion.  
* **Product Realism:** Given that product linking is to *mock* URLs based on generic categories or simple searches, the visual appearance of the AI-generated item might not perfectly match a specific product's design, but it will adhere to the specified style and type. The system aims for *representative* suggestions.