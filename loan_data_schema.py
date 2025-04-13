from pydantic import BaseModel
from typing import List

class LoanApplication(BaseModel):
    features: List[float]  # Same order as training features
