from toolbox import generate_data, evaluate
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from mlens.ensemble import SuperLearner
from mlens.metrics.metrics import rmse
from mlens.utils import pickle_save
from cuml import CD, RandomForestRegressor, LinearRegression, Lasso, ElasticNet, Ridge, KNeighborsRegressor
from joblib import Memory
import cupy
import rmm

# Manage memory
cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
rmm.reinitialize(pool_allocator=True, managed_memory=True)
location = './cachedir'
memory = Memory(location, verbose=0)

# Set seed
seed = 10

# Generate data and cache result
generate_data = memory.cache(generate_data)
X, X_val, y, y_val = generate_data(lat_min=30, lat_max=150, lon_min=0, lon_max=360, seed=10, impute=True, impute_method='simple')

# Define model
estimators = [
    LGBMRegressor(random_state=seed, device='GPU'),
    XGBRegressor(random_state=seed, tree_method='gpu_hist', use_rmm=True, n_estimators=4000),
    CatBoostRegressor(random_state=seed, verbose=0, task_type="GPU", n_estimators=2000, max_depth=16),
    RandomForestRegressor(n_estimators=100, n_bins=256, max_depth=16),
    KNeighborsRegressor(),
    ]

meta_est = [XGBRegressor(random_state=seed, tree_method='gpu_hist', use_rmm=True, n_estimators=4000)]

# Create ensemble
ensemble = SuperLearner(scorer=rmse, random_state=seed, verbose=0, backend='sequential')
ensemble.add(estimators=estimators, propagate_features=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
ensemble.add_meta(estimator=meta_est)

# Fit ensemble
print("Fitting model")
ensemble.fit(X, y)

# Save model
print('Saving model')
pickle_save(ensemble, 'results/ensemble.pkl')

# Evaluate model
evaluate(ensemble, X_val, y_val)
