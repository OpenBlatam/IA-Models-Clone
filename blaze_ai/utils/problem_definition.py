from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricSpec:
    name: str
    greater_is_better: bool = True
    threshold: Optional[float] = None


@dataclass
class DatasetField:
    name: str
    dtype: str
    description: str = ""
    is_target: bool = False
    is_categorical: bool = False
    allow_missing: bool = True


@dataclass
class DatasetSchema:
    name: str
    fields: List[DatasetField] = field(default_factory=list)
    path: Optional[str] = None
    description: str = ""

    def target_field(self) -> Optional[DatasetField]:
        for f in self.fields:
            if f.is_target:
                return f
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "fields": [f.__dict__ for f in self.fields],
        }


@dataclass
class ProblemDefinition:
    title: str
    objective: str
    inputs: List[str]
    outputs: List[str]
    dataset: DatasetSchema
    metrics: List[MetricSpec] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.title or not self.objective:
            raise ValueError("title and objective are required")
        if not self.inputs or not self.outputs:
            raise ValueError("inputs and outputs cannot be empty")
        if self.dataset.target_field() is None:
            raise ValueError("dataset must define a target field")
        for m in self.metrics:
            if not m.name:
                raise ValueError("metric name cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "objective": self.objective,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dataset": self.dataset.to_dict(),
            "metrics": [m.__dict__ for m in self.metrics],
            "constraints": self.constraints,
            "assumptions": self.assumptions,
        }


