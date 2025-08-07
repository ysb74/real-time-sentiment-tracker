"""
Image sentiment analysis via CLIP/DINOv2 or custom model.
"""

import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

class ImageSentimentAnalyzer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # Alternatively, for DINOv2: use facebookresearch/dinov2.

    def analyze(self, image_path_or_url):
        if image_path_or_url.startswith("http"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw)
        else:
            image = Image.open(image_path_or_url)
        # Pseudocode: image-to-sentiment via prompt engineering
        texts = ["a happy photo", "a sad photo", "a neutral photo"]
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image.softmax(dim=1).squeeze()
        idx = logits_per_image.argmax().item()
        classes = ["positive", "negative", "neutral"]
        return {"label": classes[idx], "confidence": float(logits_per_image[idx])}
