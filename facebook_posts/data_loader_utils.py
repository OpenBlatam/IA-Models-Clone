from typing import Optional, Callable
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    *,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    generator_seed: Optional[int] = None,
    sampler: Optional[Sampler] = None,
    batch_sampler: Optional[BatchSampler] = None,
) -> DataLoader:
    """Create a performant DataLoader with sensible defaults.

    - Uses pin_memory when CUDA is available
    - Enables persistent workers for throughput
    - Allows deterministic seeding via generator_seed
    - Supports external sampler/batch_sampler for DDP or CV folds
    """
    resolved_num_workers = num_workers if num_workers is not None else max(1, torch.get_num_threads() // 2)

    generator = None
    if generator_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(generator_seed)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None and batch_sampler is None else False,
        num_workers=resolved_num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers if resolved_num_workers > 0 else False,
        prefetch_factor=prefetch_factor if resolved_num_workers > 0 else 2,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
        batch_sampler=batch_sampler,
        generator=generator,
    )

    return dataloader







