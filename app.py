import os
import base64
import random
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from datasets import load_dataset
import torch
import torch.nn as nn
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 1. CONFIGURATION & CONSTANTS
# ==========================================
MODEL_FILENAME = 'best_captcha_crnn.pth' 

RNN_HIDDEN_SIZE = 256  
RNN_LAYERS = 2         

# The EXACT string from your Colab notebook
RAW_CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_LIST = sorted(list(RAW_CHARACTERS))

# Shift everything up by 1 so Index 0 can be the Blank Token
NUM_TO_CHAR = {i + 1: char for i, char in enumerate(CHAR_LIST)}
BLANK_INDEX = 0 
NUM_TO_CHAR[BLANK_INDEX] = ""

# ==========================================
# ☁️ 2. LOAD HUGGING FACE DATASET
# ==========================================
print("☁️ Connecting to Hugging Face dataset...")
try:
    hf_dataset = load_dataset("hammer888/captcha-data", split="train")
    
    # [Alignment] Apply the exact same 1000-image subset used in evaluate.py
    hf_dataset = hf_dataset.select(range(1000))
    
    print(f"✅ Successfully loaded {len(hf_dataset)} images from the cloud!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    hf_dataset = None

# ==========================================
# 🏗️ 3. YOUR CRNN MODEL CLASS
# ==========================================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(1536, RNN_HIDDEN_SIZE, bidirectional=True, num_layers=RNN_LAYERS, batch_first=False)
        self.fc = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        features = self.cnn(x) 
        b, c, h, w = features.size()
        features = features.view(b, c * h, w) 
        features = features.permute(2, 0, 1)  

        out, _ = self.rnn(features)
        out = self.fc(out) 
        return out

# ==========================================
# 🧠 4. INITIALIZE AND LOAD WEIGHTS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing model on {device}...")

model = CRNN(num_classes=len(CHAR_LIST)).to(device)

if not os.path.exists(MODEL_FILENAME):
    print(f"❌ ERROR: I cannot find {MODEL_FILENAME}! Please put it in this folder.")
else:
    try:
        model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device, weights_only=False))
        model.eval()
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading weights into architecture: {e}")

# ==========================================
# 🖼️ 5. IMAGE PREPROCESSING PIPELINE
# ==========================================
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((50, 200)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

# ==========================================
# 🛠️ 6. CTC GREEDY DECODER
# ==========================================
def ctc_greedy_decode(outputs):
    outputs = outputs.squeeze(1) 
    _, max_indices = torch.max(outputs, dim=1)
    max_indices = max_indices.tolist()
    
    predicted_text = ""
    prev_idx = -1
    
    for idx in max_indices:
        if idx != prev_idx:  
            if idx != BLANK_INDEX: 
                predicted_text += NUM_TO_CHAR.get(idx, "")
        prev_idx = idx
        
    return predicted_text

# ==========================================
# 🌐 7. FLASK APP ROUTES
# ==========================================
app = Flask(__name__)

def predict_captcha_with_ai(image_bytes):
    try:
        # [Alignment] Exact identical image processing to evaluate.py
        image = Image.open(BytesIO(image_bytes)).convert('L')
        tensor = preprocess(image).unsqueeze(0).to(device) 
        
        with torch.no_grad():
            outputs = model(tensor) 
            
        predicted_text = ctc_greedy_decode(outputs)
        
        return predicted_text if predicted_text != "" else "[Blank Prediction]"
        
    except Exception as e:
        return f"Prediction Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['GET'])
def generate_captcha():
    if hf_dataset is None:
        return jsonify({'error': 'Dataset not loaded. Check your internet connection or HF token.'}), 500
        
    random_idx = random.randint(0, len(hf_dataset) - 1)
    item = hf_dataset[random_idx]
    
    image = item['image']
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # [Alignment] Exact identical ground truth extraction from evaluate.py
    raw_sentence = item['text']
    if 'text' in raw_sentence:
        actual_text = raw_sentence.split('text')[-1].strip().replace(" ", "").replace("'", "").replace('"', "")
    else:
        actual_text = raw_sentence.strip().replace(" ", "").replace("'", "").replace('"', "")
    
    return jsonify({
        'image': f"data:image/png;base64,{base64_img}",
        'actual_text': actual_text 
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    image_b64 = data.get('image')
    
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400
        
    if "," in image_b64:
        header, encoded = image_b64.split(",", 1)
    else:
        encoded = image_b64
        
    image_bytes = base64.b64decode(encoded)
    
    prediction = predict_captcha_with_ai(image_bytes)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=False)