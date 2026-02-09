"""Data Processing Utilities"""

def generate_data_processing_code() -> str:
    return '''"""
Data Processing
===============

Utilidades para procesamiento de datos.
"""

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict
import torch
from torchvision import transforms
from PIL import Image


class TextDataset(Dataset):
    """Dataset para texto."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class ImageDataset(Dataset):
    """Dataset para imágenes."""
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return {'image': self.transform(image), 'path': self.image_paths[idx]}
'''

