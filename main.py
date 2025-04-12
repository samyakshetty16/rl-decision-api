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
