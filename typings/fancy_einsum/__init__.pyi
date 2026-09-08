# This project uses only the PyTorch backend of fancy-einsum.
from torch import Tensor

def einsum(equation: str, *operands: Tensor) -> Tensor: ...
