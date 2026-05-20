"""
Tokenization Utils - Utilidades de Tokenización Avanzada
==========================================================

Utilidades para tokenización y procesamiento de texto.
"""

import logging
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Callable
import torch

logger = logging.getLogger(__name__)

# Intentar importar transformers
try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    _has_transformers = True
except ImportError:
    _has_transformers = False
    logger.warning("transformers not available, some tokenization functions will be limited")


class TextPreprocessor:
    """
    Preprocesador de texto con múltiples opciones.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_accents: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_whitespace: bool = False,
        normalize_whitespace: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False
    ):
        """
        Inicializar preprocesador.
        
        Args:
            lowercase: Convertir a minúsculas
            remove_accents: Remover acentos
            remove_punctuation: Remover puntuación
            remove_numbers: Remover números
            remove_whitespace: Remover espacios en blanco
            normalize_whitespace: Normalizar espacios
            remove_urls: Remover URLs
            remove_emails: Remover emails
            remove_mentions: Remover menciones (@)
            remove_hashtags: Remover hashtags (#)
        """
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_whitespace = remove_whitespace
        self.normalize_whitespace = normalize_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
    
    def preprocess(self, text: str) -> str:
        """
        Preprocesar texto.
        
        Args:
            text: Texto original
            
        Returns:
            Texto preprocesado
        """
        if not text:
            return ""
        
        # Remover URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remover emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remover menciones
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remover hashtags
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Convertir a minúsculas
        if self.lowercase:
            text = text.lower()
        
        # Remover acentos
        if self.remove_accents:
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Remover puntuación
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remover números
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Normalizar espacios
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        # Remover espacios
        if self.remove_whitespace:
            text = text.replace(' ', '')
        
        return text.strip()


class AdvancedTokenizer:
    """
    Tokenizador avanzado con múltiples opciones.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        """
        Inicializar tokenizador.
        
        Args:
            tokenizer_name: Nombre del tokenizador
            max_length: Longitud máxima
            padding: Aplicar padding
            truncation: Aplicar truncation
            return_tensors: Formato de retorno
        """
        if not _has_transformers:
            raise ImportError("transformers is required for AdvancedTokenizer")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
    
    def tokenize(
        self,
        texts: List[str],
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizar textos.
        
        Args:
            texts: Lista de textos
            return_attention_mask: Retornar attention mask
            return_token_type_ids: Retornar token type IDs
            
        Returns:
            Diccionario con tokens
        """
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids
        )
        return encodings
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decodificar tokens a texto.
        
        Args:
            token_ids: IDs de tokens
            skip_special_tokens: Saltar tokens especiales
            
        Returns:
            Lista de textos decodificados
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], list):
            return [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]
        else:
            return [self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)]


class DynamicPadding:
    """
    Padding dinámico para batches.
    """
    
    def __init__(self, pad_token_id: int = 0):
        """
        Inicializar dynamic padding.
        
        Args:
            pad_token_id: ID del token de padding
        """
        self.pad_token_id = pad_token_id
    
    def pad_batch(
        self,
        sequences: List[torch.Tensor],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Aplicar padding a un batch.
        
        Args:
            sequences: Lista de secuencias
            max_length: Longitud máxima (opcional)
            
        Returns:
            Tensor con padding
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            padding_length = max_length - len(seq)
            if padding_length > 0:
                padding = torch.full((padding_length,), self.pad_token_id, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq[:max_length]
            padded.append(padded_seq)
        
        return torch.stack(padded)


class TokenizerWrapper:
    """
    Wrapper para tokenizadores con preprocesamiento.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        preprocessor: Optional[TextPreprocessor] = None,
        max_length: int = 512
    ):
        """
        Inicializar wrapper.
        
        Args:
            tokenizer: Tokenizador de transformers
            preprocessor: Preprocesador opcional
            max_length: Longitud máxima
        """
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def __call__(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenizar textos con preprocesamiento.
        
        Args:
            texts: Lista de textos
            **kwargs: Argumentos adicionales para tokenizer
            
        Returns:
            Diccionario con tokens
        """
        # Preprocesar si hay preprocesador
        if self.preprocessor:
            texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Tokenizar
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs
        )


def create_tokenizer(
    model_name: str,
    preprocessor_config: Optional[Dict] = None
) -> TokenizerWrapper:
    """
    Crear tokenizador con preprocesador.
    
    Args:
        model_name: Nombre del modelo
        preprocessor_config: Configuración del preprocesador
        
    Returns:
        TokenizerWrapper
    """
    if not _has_transformers:
        raise ImportError("transformers is required")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preprocessor = TextPreprocessor(**(preprocessor_config or {}))
    
    return TokenizerWrapper(tokenizer, preprocessor)




