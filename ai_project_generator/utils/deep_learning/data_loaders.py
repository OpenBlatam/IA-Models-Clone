"""
Data Loaders
============

Utilidades para carga y procesamiento de datos.
"""

from typing import Dict, List, Any, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset para datos de texto."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ):
        """
        Args:
            texts: Lista de textos
            tokenizer: Tokenizer de transformers
            max_length: Longitud máxima de secuencia
            padding: Si hacer padding
            truncation: Si truncar secuencias largas
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length' if self.padding else False,
            truncation=self.truncation,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # Para language modeling
        }


class ImageDataset(Dataset):
    """Dataset para datos de imágenes."""
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Args:
            image_paths: Lista de paths a imágenes
            transform: Transformaciones para imágenes
            target_transform: Transformaciones para targets
        """
        self.image_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from PIL import Image
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'path': image_path}


class DataLoaderFactory:
    """Factory para crear DataLoaders."""
    
    @staticmethod
    def create_text_dataloader(
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        max_length: int = 512,
        **kwargs
    ) -> DataLoader:
        """Crea DataLoader para texto."""
        dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )
    
    @staticmethod
    def create_image_dataloader(
        image_paths: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        transform: Optional[Callable] = None,
        **kwargs
    ) -> DataLoader:
        """Crea DataLoader para imágenes."""
        dataset = ImageDataset(
            image_paths=image_paths,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )


def generate_data_loader_code(data_type: str = "text") -> str:
    """Genera código para data loaders."""
    if data_type == "text":
        return '''"""
Data Loaders for Text Data
===========================
"""

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict
import torch


class TextDataset(Dataset):
    """Dataset para datos de texto."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length' if self.padding else False,
            truncation=self.truncation,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def create_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    max_length: int = 512
) -> DataLoader:
    """Crea DataLoader para texto."""
    dataset = TextDataset(texts, tokenizer, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
'''
    else:
        return '''"""
Data Loaders for Image Data
===========================
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Optional, Callable
import torch


class ImageDataset(Dataset):
    """Dataset para datos de imágenes."""
    
    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'path': image_path}


def create_dataloader(
    image_paths: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224
) -> DataLoader:
    """Crea DataLoader para imágenes."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(image_paths, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
'''

