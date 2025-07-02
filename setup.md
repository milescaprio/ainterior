# Step 1: Keys
- Create a file called keys.py and copy the following code. Enter your keys in the following variables:

```
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # <- Enter your Gemini API Key here
REPLICATE_API_TOKEN = "YOUR_REPLICATE_API_TOKEN" <- Enter your Replicate API Key here
```

# Step 2: Terminal Setup
- Open a terminal window in this directory and run the following:
```
conda env create
conda activate ainterior
```

// For Mac silicon:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

// For Windows Cuda: 
```
pip install torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu118 
```

```
pip install ultralytics-thop
```

```
streamlit run main.py
```