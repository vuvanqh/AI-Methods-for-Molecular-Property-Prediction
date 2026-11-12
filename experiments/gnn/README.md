# Multi-Task GNN Training

Multi-task GNN training and MLflow setup for the 18 endpoints (`12 Tox21`, `5 CYP450_Veith`, `1 hERG_Karim`).

## Usage

# Full training run (automatically runs gnn_data/preprocess.py on first run)
uv run python train.py

# Launch local MLflow UI (http://127.0.0.1:5000)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
