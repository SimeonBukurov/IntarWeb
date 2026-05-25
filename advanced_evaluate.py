import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
import difflib
import collections
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 1. CONFIGURATION & CONSTANTS
# ==========================================
MODEL_FILENAME = 'best_captcha_crnn.pth' 

RAW_CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_LIST = sorted(list(RAW_CHARACTERS))
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
# 🔬 3. ADVANCED ERROR ANALYSIS LOGIC
# ==========================================
def analyze_errors(true_str, pred_str):
    """Uses Python's difflib to align strings and count specific error types."""
    subs = 0
    dels = 0
    ins = 0
    confusions = [] 

    matcher = difflib.SequenceMatcher(None, true_str, pred_str)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Align the replaced blocks
            t_sub = true_str[i1:i2]
            p_sub = pred_str[j1:j2]
            length = max(len(t_sub), len(p_sub))
            t_sub = t_sub.ljust(length, '-')
            p_sub = p_sub.ljust(length, '-')
            
            for t_c, p_c in zip(t_sub, p_sub):
                if t_c == '-': ins += 1
                elif p_c == '-': dels += 1
                else:
                    subs += 1
                    confusions.append((t_c, p_c))
        elif tag == 'delete':
            dels += (i2 - i1)
        elif tag == 'insert':
            ins += (j2 - j1)

    return subs, dels, ins, confusions

# ==========================================
# 🚀 4. CLOUD EVALUATION LOOP
# ==========================================
def run_detailed_evaluation():
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
        dataset = load_dataset("hammer888/captcha-data", split="train")
        # Restrict to the exact same 1000 image baseline
        test_subset = dataset.select(range(1000))
    except Exception as e:
        print(f"❌ Failed to download from Hugging Face: {e}")
        return

    print(f"Starting detailed evaluation on {len(test_subset)} cloud images...\n")
    
    total_captchas = 0
    correct_exact_matches = 0
    total_characters = 0
    
    # Advanced metrics trackers
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    confusion_tracker = collections.Counter()
    all_mistakes = []

    with torch.no_grad():
        for i, item in enumerate(test_subset):
            # 1. Bulletproof ground truth extraction (no more weird spaces or quotes)
            raw_sentence = item['text']
            clean_sentence = raw_sentence.replace("'", "").replace('"', "")
            true_label = clean_sentence.split()[-1]
            
            # 2. Get the cloud image
            image = item['image'].convert('L')
            
            tensor_image = preprocess(image).unsqueeze(0).to(device)
            predicted_label = ctc_greedy_decode(model(tensor_image))
            
            # Metrics
            total_captchas += 1
            total_characters += len(true_label)
            
            if predicted_label == true_label:
                correct_exact_matches += 1
            else:
                subs, dels, ins, confusions = analyze_errors(true_label, predicted_label)
                total_substitutions += subs
                total_deletions += dels
                total_insertions += ins
                confusion_tracker.update(confusions)
                
                # Save to full report
                all_mistakes.append(f"Index: [{i:<4}] Actual: {true_label:<10} | Predicted: {predicted_label:<10}")
                
            if total_captchas % 200 == 0:
                print(f"Processed {total_captchas}/{len(test_subset)}...")

    # Final Math
    exact_match_accuracy = (correct_exact_matches / total_captchas) * 100 if total_captchas > 0 else 0
    total_errors = total_substitutions + total_deletions + total_insertions
    cer = (total_errors / total_characters) * 100 if total_characters > 0 else 0
    
    # Print Dashboard
    print("\n" + "="*50)
    print("📊 ADVANCED EVALUATION DASHBOARD")
    print("="*50)
    print(f"Total CAPTCHAs Tested: {total_captchas}")
    print(f"Total Characters Read: {total_characters}")
    print(f"Exact Match Accuracy:  {exact_match_accuracy:.2f}%")
    print(f"Character Error Rate:  {cer:.2f}%\n")
    
    print("🔍 BREAKDOWN OF CHARACTER ERRORS:")
    print(f" - Substitutions (Wrong Letter): {total_substitutions}")
    print(f" - Deletions     (Missed Letter): {total_deletions}")
    print(f" - Insertions    (Extra Letter):  {total_insertions}\n")
    
    print("⚠️ TOP 10 MOST COMMON CONFUSIONS (Actual -> Predicted):")
    if not confusion_tracker:
        print("  None! The model is perfect.")
    else:
        for (true_c, pred_c), count in confusion_tracker.most_common(10):
            print(f"  '{true_c}' was read as '{pred_c}' : {count} times")
            
    # Save the huge list of mistakes to a text file
    if all_mistakes:
        report_path = "detailed_mistakes.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Total Mistakes: {len(all_mistakes)}\n")
            f.write("="*40 + "\n")
            f.write("\n".join(all_mistakes))
        print(f"\n📂 A full list of every single mistake has been saved to: {report_path}")

if __name__ == "__main__":
    run_detailed_evaluation()