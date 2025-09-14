"""
Definition of pydantic classes to make impunt validation.
"""

from pydantic import BaseModel, PositiveInt, NonNegativeInt , FilePath, Field, ValidationInfo, field_validator
from typing import Tuple, List, Dict, Union, Sequence, Optional
import yaml
import warnings


def parse_ranges(s: str):
    """
    Converts strings like "1-5, 9-22, 30" into a list of integers.
    """
    result = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
            except ValueError:
                raise ValueError(f"Invalid range format: '{part}' (expected e.g. 1-5)")
            if start > end:
                raise ValueError(f"Invalid range order: '{part}' (start > end)")
            result.extend(range(start, end + 1))
        else:
            try:
                result.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid value: '{part}' (expected a number or range)")
    return result

class ModelConfig(BaseModel):
    name: str
    epochs: int = Field(..., ge=0)
    batch_size: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    early_stop_patience: int = Field(..., ge=0)
    lr_plateau_reduction_patience: int = Field(..., gt=0)
    validation_size: float = Field(..., gt=0, lt=1)
    use_features: Union[List[PositiveInt]]
    conv_layers: Optional[List[Tuple[PositiveInt, PositiveInt]]] = None
    conv_pool_size: Optional[List[NonNegativeInt]] = None
    hidden_layers: List[PositiveInt]
    dropout_layers: List[float]

    @field_validator("epochs")
    def warn_if_zero_epochs(cls, v, info: ValidationInfo):
        """
        Verify if the ephocs is given equal to 0, if so rise a warning.
        """
        if v == 0:
            warnings.warn(
                f"\033[38;5;208m[INPUT WARNING]\033[0m '{info.data.get('name', '?')}' will not be trained (epochs=0). Default weights will be used.",
                UserWarning
            )
        return v

    @field_validator("use_features", mode="before")
    def parse_use_features(cls, v):
        """
        Parse the use-features and rise an error if contain the colun 0 which must be the label column.
        """
        if isinstance(v, str):
            v = parse_ranges(v)
        if isinstance(v, (list, tuple)):
            if 0 in v:
                raise ValueError("Column 0 is the target y; 0 cannot be included in the feature indices")
            v = list(dict.fromkeys(v))
        return v
    
    @field_validator("conv_pool_size")
    def validate_and_pad_conv_pool_size(cls, v, info: ValidationInfo):
        """
        Verify if the conv_pool_size has the correct length.
        """
        conv_layers = info.data.get("conv_layers")
        use_features = info.data.get("use_features")
        expected_len = len(conv_layers)

        if conv_layers is None:
            return None
        if v is None:
            return [0] * expected_len
        
        if len(v) != expected_len:
            raise ValueError(f"conv_pool_size length ({len(v)}) must be equal to conv_layers length ({expected_len})")
        
        num_features = len(use_features)
        for pool_size in v:
            if pool_size > num_features:
                raise ValueError(f"Each conv_pool_size must be less or equal to the number of features ({num_features})")
        return v

    @field_validator("dropout_layers")
    def check_dropout(cls, v, info: ValidationInfo):
        """
        Verify if the dropout matches the length of the hidden_layers.
        """
        if not all(0 <= x < 1 for x in v):
            raise ValueError("All dropout_layers values must be greater than or equal to 0 and less than 1.")
        hidden_layers = info.data.get("hidden_layers")
        if hidden_layers is not None and len(v) != len(hidden_layers):
            raise ValueError(
                f"dropout_layers and hidden_layers must have the same length "
                f"(expected {len(hidden_layers)}, got {len(v)})"
            )
        return v  

class GlobalConfig(BaseModel):
    input_file_path: FilePath
    test_size: float = Field(..., gt=0, lt=1)
    seed: int = 42
    class_labels: Tuple[str, str]
    model_parameters: List[ModelConfig]

def load_config(path="config.yaml") -> GlobalConfig:
    """
    Reads the config.yaml
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    return GlobalConfig(**data)
