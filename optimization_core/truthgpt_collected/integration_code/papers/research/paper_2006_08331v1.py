import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

class Paper_2006_08331v1Config:
    """
    Configuration for the probing module designed to understand what neural dialog models
    encode about conversational properties (e.g., dialogue acts, sentiment, topic shifts).

    Attributes:
        hidden_dim (int): Dimensionality of the hidden states from the backbone dialog model.
        num_labels_per_task (List[int]): Number of output classes for each probing task.
        dropout (float): Dropout probability applied between projection layers.
        use_mlp (bool): If True, use a two-layer MLP probe; otherwise use a linear probe.
        temperature (float): Temperature for scaling logits (useful for calibration).
        shared_representation (bool): If True, share a single hidden representation across tasks.
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        num_labels_per_task: List[int] = [7, 3, 5],  # e.g., dialogue acts, sentiment, domain
        dropout: float = 0.1,
        use_mlp: bool = False,
        temperature: float = 1.0,
        shared_representation: bool = True,
    ):
        self.hidden_dim = hidden_dim
        self.num_labels_per_task = num_labels_per_task
        self.dropout = dropout
        self.use_mlp = use_mlp
        self.temperature = temperature
        self.shared_representation = shared_representation

    def __repr__(self):
        return (f"Paper_2006_08331v1Config(hidden_dim={self.hidden_dim}, "
                f"num_labels_per_task={self.num_labels_per_task}, "
                f"dropout={self.dropout}, use_mlp={self.use_mlp}, "
                f"temperature={self.temperature}, shared_rep={self.shared_representation})")


class Paper_2006_08331v1Module(nn.Module):
    """
    Probing module for neural dialog models.
    Implements the methodology described in "Probing Neural Dialog Models for Conversational Understanding":
    Attach diagnostic classifiers to hidden states of a frozen dialog model to evaluate what
    conversational properties (e.g., dialogue acts, sentiment, topic transitions) are encoded.

    This module supports multi-task probing, where multiple classifiers share a common
    representation or use independent projections. The probing can be linear or a shallow MLP,
    following standard probing literature (Belinkov et al., 2019; Tenney et al., 2019).

    Usage:
        1. Extract hidden states (e.g., last layer) from a pretrained dialog model (e.g., DialoGPT).
        2. Feed the representation (per utterance or context) to this module.
        3. Train the probe parameters on labeled probing data while keeping the backbone frozen.

    References:
        - Belinkov et al., "What do you learn from context? Probing for sentence structure in
          contextualized word representations." ICLR 2019.
        - Tenney et al., "BERT rediscovers the classical NLP pipeline." ACL 2019.
    """
    def __init__(self, config: Paper_2006_08331v1Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_tasks = len(config.num_labels_per_task)
        self.dropout = nn.Dropout(config.dropout)
        self.temperature = config.temperature

        # Shared representation project (optional)
        if config.shared_representation:
            if config.use_mlp:
                self.shared_proj = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
                rep_dim = config.hidden_dim
            else:
                # No non-linear projection; use raw hidden state as shared rep
                self.shared_proj = nn.Identity()
                rep_dim = config.hidden_dim
        else:
            self.shared_proj = nn.Identity()
            rep_dim = config.hidden_dim

        # Task-specific output heads
        self.task_heads = nn.ModuleList()
        for num_labels in config.num_labels_per_task:
            if config.use_mlp and config.shared_representation:
                # Shared MLP already applied; just a linear classifier
                head = nn.Linear(rep_dim, num_labels)
            elif config.use_mlp and not config.shared_representation:
                # Individual MLP per task
                head = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, num_labels),
                )
            else:
                # Linear probe (default)
                head = nn.Linear(rep_dim if config.shared_representation else config.hidden_dim,
                                 num_labels)
            self.task_heads.append(head)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        task_ids: Optional[List[int]] = None,
        return_representations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states (Tensor): Shape (batch_size, hidden_dim) – representation from dialog model.
            task_ids (List[int], optional): Indices of tasks to compute. If None, all tasks.
            return_representations (bool): If True, also return intermediate shared representation.

        Returns:
            dict: 'logits' – list of tensors, each shape (batch_size, num_labels_i).
                  'representations' – (optional) shared representation if requested.
        """
        if hidden_states.dim() == 3:
            # If input is (batch, seq_len, hidden_dim), mean-pool over sequence
            hidden_states = hidden_states.mean(dim=1)

        # Apply shared projection (if any)
        shared_rep = self.shared_proj(hidden_states)
        shared_rep = self.dropout(shared_rep)

        task_logits = []
        tasks_to_compute = task_ids if task_ids is not None else list(range(self.num_tasks))
        for i in tasks_to_compute:
            head = self.task_heads[i]
            logits = head(shared_rep if self.config.shared_representation else hidden_states)
            # Temperature scaling
            logits = logits / self.temperature
            task_logits.append(logits)

        out = {"logits": task_logits}
        if return_representations:
            out["representations"] = shared_rep
        return out

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: List[torch.Tensor],
        task_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute multi-task cross-entropy loss.

        Args:
            hidden_states (Tensor): (batch_size, hidden_dim).
            labels (List[Tensor]): List of label tensors, each (batch_size,). Length must match
                                   the number of tasks or task_ids.
            task_ids (List[int], optional): Which tasks to compute loss for.

        Returns:
            Tensor: Scalar loss (summed over tasks).
        """
        outputs = self.forward(hidden_states, task_ids=task_ids)
        logits_list = outputs["logits"]
        loss = 0.0
        for logits, lbl in zip(logits_list, labels):
            loss += F.cross_entropy(logits, lbl)
        return loss


# =============================================================================
# Demonstration: Full testing with a mock dialog encoder
# =============================================================================
if __name__ == "__main__":
    # Simulate a dialog model (e.g., a small transformer) that outputs hidden states
    class MockDialogEncoder(nn.Module):
        def __init__(self, hidden_dim=256, num_layers=2, vocab_size=1000):
            super().__init__()
            from transformers import AutoConfig, AutoModel  # optional, but for simplicity:
            # We'll use a simple embedding + LSTM as a placeholder
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.hidden_dim = hidden_dim

        def forward(self, input_ids):
            # input_ids: (batch, seq_len)
            emb = self.embed(input_ids)
            h, _ = self.lstm(emb)  # (batch, seq_len, hidden_dim)
            return h  # return all hidden states

    # Configuration for probing (e.g., 3 tasks: dialogue act, sentiment, topic)
    config = Paper_2006_08331v1Config(
        hidden_dim=256,
        num_labels_per_task=[5, 3, 4],  # e.g., 5 act types, 3 sentiments, 4 topics
        dropout=0.2,
        use_mlp=False,      # linear probe (most common in probing literature)
        temperature=1.0,
        shared_representation=True,
    )

    # Initialize encoder (frozen, typically)
    encoder = MockDialogEncoder(hidden_dim=config.hidden_dim)
    for param in encoder.parameters():
        param.requires_grad = False  # freeze encoder

    # Initialize probing module
    probe = Paper_2006_08331v1Module(config)

    # Create dummy batch
    batch_size = 4
    seq_len = 10
    vocab_size = 1000
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Labels for three tasks
    labels = [
        torch.randint(0, 5, (batch_size,)),  # task 0
        torch.randint(0, 3, (batch_size,)),  # task 1
        torch.randint(0, 4, (batch_size,)),  # task 2
    ]

    # Forward pass through encoder
    with torch.no_grad():
        hidden_states = encoder(input_ids)  # (batch, seq_len, hidden_dim)

    # Probe forward
    outputs = probe(hidden_states, return_representations=True)
    logits_list = outputs["logits"]
    rep = outputs["representations"]
    print("Probe logits shapes:", [l.shape for l in logits_list])
    print("Shared representation shape:", rep.shape)

    # Compute loss
    loss = probe.compute_loss(hidden_states, labels)
    print("Probe loss:", loss.item())

    # Backward optimization (simulate one step)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Training step successful.")

    # Show that mode is ready for production probing tasks
    print("\nProbing module ready for attachment to any decoder-based dialog model.")
    print("To use: extract hidden states from the last layer, then feed to this module.")