# rl-decision-api

This API uses a trained Reinforcement Learning (RL) model to return loan approval decisions based on:

- Borrower features
- Macroeconomic indicators
- XGBoost model predictions

## Run Locally

```bash
uvicorn main:app --reload
