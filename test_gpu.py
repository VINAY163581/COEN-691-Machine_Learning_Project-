import xgboost as xgb
import numpy as np

print("XGBoost version:", xgb.__version__)

# Create some sample data
data = np.random.rand(5, 5)
label = np.random.randint(2, size=5)

# Create DMatrix
dtrain = xgb.DMatrix(data, label=label)

# Set parameters for GPU training (using new API)
param = {
    'tree_method': 'hist',
    'device': 'cuda'
}

try:
    # Try to train a model using GPU
    bst = xgb.train(param, dtrain, num_boost_round=1)
    print('XGBoost GPU support is working!')
    print('Using device:', param['device'])
except Exception as e:
    print('Error:', str(e))