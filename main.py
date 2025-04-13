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
'''

'''
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import DQN

app = FastAPI()

# Load the RL model (ensure this path is correct)
model = DQN.load("rl_models/dqn_model.zip", device="cpu")

# Define the input data schema
class StateInput(BaseModel):
    borrower_features: list  # Should contain 9 features
    macro_features: list     # Should contain 14 features
    xgb_predictions: list    # Should contain 2 features

@app.post("/rl_decision")
def get_rl_action(input_data: StateInput):
    # Combine borrower features, macroeconomic features, and XGBoost predictions
    state = input_data.borrower_features + input_data.macro_features + input_data.xgb_predictions
    
    # Ensure the state has exactly 34 features
    if len(state) != 34:
        return {"error": "Invalid input: Expected 34 features, got " + str(len(state))}
    
    # Convert state to numpy array
    state = np.array(state).reshape(1, -1)
    
    # Get the RL model prediction (deterministic action)
    action = model.predict(state, deterministic=True)[0]
    
    # Map actions to the decision
    action_map = {
        0: "Deny",
        1: "Approve",
        2: "Adjust terms"
    }
    
    return {"rl_decision": action_map.get(int(action), "Unknown")}
'''





from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import torch

from stable_baselines3 import DQN

app = FastAPI(title="RL Loan Decision API")

# Load the trained DQN model
dqn_model = DQN.load("rl_models/dqn_model.zip")  # adjust the path as needed

# Final features expected by RL model (in exact order)
feature_names = [
    'loan_purpose_all_other', 'loan_purpose_credit_card', 'loan_purpose_debt_consolidation',
    'loan_purpose_educational', 'loan_purpose_home_improvement', 'loan_purpose_major_purchase',
    'loan_purpose_small_business',
    'interest_rate', 'monthly_installment', 'log_annual_income', 'debt_to_income',
    'credit_line_days', 'revolving_balance', 'revolving_utilization',
    'inquiries_last_6mths', 'past_due_2yrs', 'public_records',
    'credit_score', 'credit_eligible',
    'GDP Growth Forecast (%)', 'CPI Inflation (%)', 'Unemployment Rate (%)',
    'Industrial Production (Index)', 'Steel Consumption (MMT)', 'Cement Production (MT)',
    'Trade Deficit ($ Billion)', 'Forex Reserves ($ Billion)', 'Bank Credit Growth (% YoY)',
    'UPI Transaction Value (? Tn)', 'Manufacturing PMI (Index)', 'Services PMI (Index)',
    'GST Collections (? Tn)', 'Fiscal Deficit (? Tn)', 'Diesel Consumption (MMT)'
]

# Expected state length
EXPECTED_FEATURE_LENGTH = 34


class RLInput(BaseModel):
    borrower_features: list  # length 17: 7 categorical + 10 numerical
    macro_features: list     # length 15
    xgb_predictions: list    # [credit_eligible, credit_score]


@app.post("/rl_decision")
def get_rl_decision(data: RLInput):
    try:
        # Combine features in correct order
        if len(data.borrower_features) != 17:
            raise ValueError(f"Expected 17 borrower features, got {len(data.borrower_features)}")

        if len(data.macro_features) != 15:
            raise ValueError(f"Expected 15 macro features, got {len(data.macro_features)}")

        if len(data.xgb_predictions) != 2:
            raise ValueError(f"Expected 2 XGB predictions, got {len(data.xgb_predictions)}")

        # credit_eligible should come first, credit_score second
        credit_eligible, credit_score = data.xgb_predictions

        # Final 34-length state vector
        state = np.array(data.borrower_features + [credit_score, credit_eligible] + data.macro_features, dtype=np.float32)

        if state.shape != (EXPECTED_FEATURE_LENGTH,):
            raise ValueError(f"Expected input vector of shape (34,), got {state.shape}")

        # Get action from DQN
        action, _ = dqn_model.predict(state, deterministic=True)

        return {
            "status": "success",
            "recommended_action": int(action),  # 0 = reject, 1 = approve
            "note": "Action based on trained RL model"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.get("/")
def root():
    return {"message": "RL Loan Decision API is up and running!"}
