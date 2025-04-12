'''
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import DQN

app = FastAPI()

# Load trained RL model
model = DQN.load("rl_models/dqn_model.zip", device="cpu")

class StateInput(BaseModel):
    borrower_features: list
    macro_features: list
    xgb_predictions: list

@app.post("/rl_decision")
def get_rl_action(input_data: StateInput):
    state = input_data.borrower_features + input_data.macro_features + input_data.xgb_predictions
    state = np.array(state).reshape(1, -1)
    action = model.predict(state, deterministic=True)[0]
    action_map = {
        0: "Deny",
        1: "Approve",
        2: "Adjust terms"
    }
    return {"rl_decision": action_map.get(int(action), "Unknown")}
'''

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from stable_baselines3 import DQN
import os

app = FastAPI()

# Load the trained RL model (safe path check)
MODEL_PATH = "rl_models/dqn_model.zip"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"RL model not found at: {MODEL_PATH}")

model = DQN.load(MODEL_PATH, device="cpu")

# Request schema using Pydantic
class StateInput(BaseModel):
    borrower_features: List[float]
    macro_features: List[float]
    xgb_predictions: List[float]

@app.post("/rl_decision")
def get_rl_action(input_data: StateInput):
    """
    Endpoint to return RL-based credit decision: Approve / Deny / Adjust terms
    """
    try:
        # Combine all input vectors into a single RL state
        state = input_data.borrower_features + input_data.macro_features + input_data.xgb_predictions
        state = np.array(state).reshape(1, -1)

        # Predict action from RL model
        action, _ = model.predict(state, deterministic=True)

        action_map = {
            0: "Deny",
            1: "Approve",
            2: "Adjust terms"
        }

        return {"rl_decision": action_map.get(int(action), "Unknown")}
    
    except Exception as e:
        return {"error": str(e)}
