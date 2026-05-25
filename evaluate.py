import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
import warnings

# Suppress some basic warnings to keep the terminal clean
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 1. CONFIGURATION & CONSTANTS
# ==========================================
MODEL_FILENAME = 'best_captcha_crnn.pth' 

# The EXACT string from your Colab notebook
RAW_CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_LIST = sorted(list(RAW_CHARACTERS))

# Shift everything up by 1 so Index 0 can be the Blank Token
NUM_TO_CHAR = {i + 1: char for i, char in enumerate(CHAR_LIST)}
BLANK_INDEX = 0 
NUM_TO_CHAR[BLANK_INDEX] = ""

# ==========================================
# 🏗️ 2. MODEL ARCHITECTURE
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
        self.rnn = nn.LSTM(1536, 256, bidirectional=True, num_layers=2, batch_first=False)
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
# 🛠️ 3. DECODER & METRICS LOGIC
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

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# ==========================================
# 🚀 4. CLOUD EVALUATION LOOP
# ==========================================
def evaluate_model():
    # Will use your GPU if you installed the CUDA version of PyTorch!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    model = CRNN(num_classes=len(CHAR_LIST)).to(device)
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device, weights_only=False))
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 200)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    print("☁️ Streaming dataset directly from Hugging Face...")
    
    try:
        # Load the dataset straight to RAM
        dataset = load_dataset("hammer888/captcha-data", split="train")
        test_subset = dataset.select(range(1000))
    except Exception as e:
        print(f"❌ Failed to download from Hugging Face: {e}")
        print("💡 Hint: You might be rate-limited. See instructions below on how to fix this.")
        return

    print(f"Starting evaluation on {len(test_subset)} images...\n")
    
    total_captchas = 0
    correct_exact_matches = 0
    total_characters = 0
    total_edits = 0
    mistakes = []

    with torch.no_grad():
        for item in test_subset:
            # 1. Clean the ground truth (e.g. "a low resolution captcha image with the text HhJ4")
            raw_sentence = item['text']
            if 'text' in raw_sentence:
                # Splits at the word 'text' and takes the actual CAPTCHA, preserving capitals
                true_label = raw_sentence.split('text')[-1].strip().replace(" ", "")
            else:
                true_label = raw_sentence.strip().replace(" ", "")
            
            # 2. Get the image directly from the cloud
            image = item['image'].convert('L')
            
            # 3. Predict via GPU
            tensor_image = preprocess(image).unsqueeze(0).to(device)
            outputs = model(tensor_image)
            predicted_label = ctc_greedy_decode(outputs)
            
            # 4. Math & Logging
            total_captchas += 1
            if predicted_label == true_label:
                correct_exact_matches += 1
            else:
                if len(mistakes) < 15: 
                    mistakes.append({'actual': true_label, 'predicted': predicted_label})
                
            total_characters += len(true_label)
            total_edits += levenshtein_distance(true_label, predicted_label)
            
            if total_captchas % 200 == 0:
                print(f"Processed {total_captchas}/{len(test_subset)}...")

    exact_match_accuracy = (correct_exact_matches / total_captchas) * 100 if total_captchas > 0 else 0
    cer = (total_edits / total_characters) * 100 if total_characters > 0 else 0
    
    print("\n" + "="*40)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Total Images Tested: {total_captchas}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    
    if mistakes:
        print("\n🔍 Snapshot of Mistakes:")
        for m in mistakes:
            print(f"Actual: {m['actual']:<8} | Predicted: {m['predicted']:<8}")

if __name__ == "__main__":
    evaluate_model()