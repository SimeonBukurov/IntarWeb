from datasets import load_dataset
import os

print("Downloading dataset from Hugging Face...")
# Load the dataset directly into memory
dataset = load_dataset("hammer888/captcha-data", split="train")

# Create the folder if it doesn't exist
save_dir = "./dataset"
os.makedirs(save_dir, exist_ok=True)

print(f"Extracting images to {save_dir}...")

# Save the first 1,000 images as PNGs (you can increase this number if you want)
# We append the index to the filename so duplicate texts don't overwrite each other
for i, item in enumerate(dataset.select(range(1000))):
    image = item['image']
    text = item['text']
    
    # Clean the text just in case there are weird characters
    clean_text = "".join(c for c in text if c.isalnum())
    
    filename = f"{clean_text}_{i}.png"
    filepath = os.path.join(save_dir, filename)
    
    image.save(filepath)

print("✅ Extraction complete! You now have actual PNG files in your folder.")