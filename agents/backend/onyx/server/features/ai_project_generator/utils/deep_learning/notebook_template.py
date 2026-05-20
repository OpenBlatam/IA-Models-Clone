"""Jupyter Notebook Template"""

def generate_notebook_template() -> str:
    return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Deep Learning Project - Exploratory Analysis\\n",
    "\\n",
    "Este notebook contiene análisis exploratorio y experimentación."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\\n",
    "from src.models import TransformerModel\\n",
    "from src.utils import set_seed, get_device\\n",
    "\\n",
    "set_seed(42)\\n",
    "device = get_device()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cargar Datos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.data import TextDataset, create_dataloader\\n",
    "\\n",
    "# Cargar datos\\n",
    "train_texts = []  # Reemplazar con tus datos\\n",
    "tokenizer = AutoTokenizer.from_pretrained('gpt2')\\n",
    "\\n",
    "train_loader = create_dataloader(train_texts, tokenizer, batch_size=32)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cargar Modelo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = TransformerModel().to(device)\\n",
    "print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualización"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot training curves\\n",
    "plt.figure(figsize=(10, 6))\\n",
    "plt.plot(train_losses, label='Train Loss')\\n",
    "plt.plot(val_losses, label='Val Loss')\\n",
    "plt.xlabel('Epoch')\\n",
    "plt.ylabel('Loss')\\n",
    "plt.legend()\\n",
    "plt.title('Training Curves')\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''

