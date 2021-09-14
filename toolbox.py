import os
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib as mpl
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

mpl.rcParams['agg.path.chunksize'] = 10000

# Function to generate data for model
def generate_data(lat_min, lat_max, lon_min, lon_max, seed, impute, impute_method):

    feats = np.load('data/MERRA2_EIS_SST_RH_Omega_U_V_Tadv_Qadv_2003_2021_monthly_global_1x1.npz')
    land_ocean_mask = np.load('data/Global_land_ocean_mask_1x1.npz')

    low_cf = np.array([])
    # day_cf = np.array([])
    # night_cf = np.array([])

    EIS = feats['EIS']
    SST = feats['SST']
    RH = feats['RH700']
    Omega = feats['W700']
    U_wind = feats['U10M']
    V_wind = feats['V10M']
    T_advection = feats['Ts_adv']
    Q_advection = feats['Qs_adv']
    Lat = feats['lat']
    Lon = feats['lon']
    year_month = np.array(np.char.split(feats['year-month'], sep='-'))
    land_mask = land_ocean_mask['land_mask']
    ocean_mask = land_ocean_mask['ocean_mask']
    cf_grids = sorted(np.load('data/low_cloud.zip').files)[219:439]
    # div10m = feats['Div10M']
    # tqv = feats['TQV']
    year = np.array([int(i[0]) for i in year_month])
    month = np.array([int(i[1]) for i in year_month])

    for grid in cf_grids:
        var = np.load('data/' + grid)
        low_cf = np.append(low_cf, var['Low_cloud_fraction'])
        # day_cf = np.append(day_cf, var['Low_cloud_fraction_Day'])
        # night_cf = np.append(night_cf, var['Low_cloud_fraction_Night'])

    low_cf = low_cf.reshape(220, 180, 360)
    # day_cf = day_cf.reshape(220, 180, 360)
    # night_cf = night_cf.reshape(220, 180, 360)

    Lat = np.tile(np.tile(np.expand_dims(Lat,  1), (1, 360)), (220, 1, 1))
    Lon = np.tile(np.tile(np.expand_dims(Lon, 1), 180).T, (220, 1, 1))
    land_mask = np.tile(np.expand_dims(land_mask, axis=0), (220, 1, 1))
    ocean_mask = np.tile(np.expand_dims(ocean_mask, axis=0), (220, 1, 1))
    month = np.tile(np.expand_dims(np.expand_dims(month, 1), 1), (1, 180, 360))
    year = np.tile(np.expand_dims(np.expand_dims(year, 1), 1), (1, 180, 360))

    total = np.stack((EIS, SST, RH, Omega, U_wind, V_wind, T_advection, Q_advection, Lat, Lon, month, year, land_mask, ocean_mask, low_cf), axis=-1)

    # Limit data between Lat and Lons
    total = np.delete(total, range(0, lat_min), axis=1)
    total = np.delete(total, range((lat_max - lat_min), lat_max), axis=1)

    total = np.delete(total, range(0, lon_min), axis=2)
    total = np.delete(total, range((lon_max - lon_min), lon_max), axis=2)

    shape = total.shape
    n_months = shape[0]
    n_lats = shape[1]
    n_lons = shape[2]
    n_feats = shape[3]

    features = np.array(np.stack(np.array_split(total, n_feats, axis=-1)[0:(n_feats - 1)], axis=3)).squeeze()
    labels = np.array(np.array_split(total, n_feats, axis=-1)[(n_feats - 1):n_feats]).squeeze()

    if impute:
        print("Imputing data")
        if impute_method == 'simple':
            imp = SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True)
        elif impute_method == 'iterative':
            imp = IterativeImputer(add_indicator=True)
        else:
            imp = impute_method

        features = features.flatten().reshape((n_months*n_lats*n_lons), (n_feats-1))
        features = imp.fit_transform(features)
        features = features.reshape(n_months, n_lats, n_lons, features.shape[1])

        X, X_val, y, y_val = train_test_split(features, labels, test_size=0.2, random_state=seed, shuffle=True)
        X_shape = X.shape
        X = X.flatten().reshape((X_shape[0] * X_shape[1] * X_shape[2]), X_shape[3])
        y = y.flatten().reshape((X_shape[0] * X_shape[1] * X_shape[2]),)
        X = np.array(X, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)
        total = (X, X_val, y, y_val)

    return total


# Write stats to text file
def stats(y_val, yhat, y_mean):
    stats = open('stats.txt', 'w+')
    stats.write('MAE %.4f' % mean_absolute_error(y_val, yhat) + '\n')
    stats.write('MAPE %.4f' % (mean_absolute_error(y_val, yhat) / y_mean) + '\n')
    stats.write('MSE %.4f' % mean_squared_error(y_val, yhat) + '\n')
    stats.write('RMSE %.4f' % (sqrt(mean_squared_error(y_val, yhat))) + '\n')
    stats.write('R2 %.4f' % + (r2_score(y_val, yhat)) + '\n')
    stats.close()

# Scatterplot
def scatterplot(data):
    sns.scatterplot(x='Actual', y='Predicted', hue='land_mask', legend='brief', data=data, palette=['b', 'g'])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Values")
    plt.savefig('scatterplot.png')
    plt.clf()

# PDF plot
def pdf_plot(data):
    sns.histplot(data=data, x='Error', kde=True, stat='density', hue='land_mask', legend='brief',
                 palette=['b', 'g'])
    plt.xlabel("Error")
    plt.ylabel("Density")
    plt.savefig('PDF.png')
    plt.clf()

# Land and ocean heatmaps
def heatmaps(ocean_data, land_data, monthly=True):

    if monthly:
        ocean_data = ocean_data[['Lat', 'Lon', 'Error']].copy()
        land_data = land_data[['Lat', 'Lon', 'Error']].copy()
        ocean_data = ocean_data.pivot(index='Lon', columns='Lat', values='Error')
        land_data = land_data.pivot(index='Lon', columns='Lat', values='Error')

    sns.heatmap(data=ocean_data, cbar=True, xticklabels='Lon', yticklabels='Lat')
    plt.title("Ocean")
    plt.savefig('heatmap_ocean.png')
    plt.clf()

    sns.heatmap(data=land_data, cbar=True, xticklabels='Lon', yticklabels='Lat')
    plt.title("Land")
    plt.savefig('heatmap_land.png')
    plt.clf()

    return ocean_data, land_data

# Evaluation function
def evaluate(ensemble, X_val_total, y_val_total):
    os.chdir('results/')
    mean_ocean_error = []
    mean_land_error = []

    def eval_helper(ensemble, X_val, y_val, mean_ocean_error=mean_ocean_error, mean_land_error=mean_land_error, monthly=True):
        # Preprocess data for graphs
        if monthly:
            shape = X_val.shape
            n_lats = shape[0]
            n_lons = shape[1]
            n_feats = shape[2]
            X_val = np.array(X_val.flatten().reshape((n_lats * n_lons), n_feats))
            y_val = np.array(y_val.flatten())
        else:
            shape = X_val.shape
            n_month = shape[0]
            n_lats = shape[1]
            n_lons = shape[2]
            n_feats = shape[3]
            X_val = np.array(X_val.flatten().reshape((n_month * n_lats * n_lons), n_feats))
            y_val = np.array(y_val.flatten())

        yhat = ensemble.predict(X_val)
        y_mean = np.mean(y_val)

        # Calculate errors for grids
        errors = np.subtract(y_val.copy(), yhat.copy())

        # Create DataFrame
        label_stack = np.stack((y_val, yhat, errors), axis=-1)
        total_data = pd.DataFrame(np.concatenate((X_val, label_stack), axis=1),
                                  columns=['EIS', 'SST', 'RH', 'Omega', 'U_wind', 'V_wind', 'T_advection', 'Q_advection', 'Lat', 'Lon', 'month', 'year', 'land_mask', 'ocean_mask', 'imp1', 'imp2', 'imp3', 'imp4', 'imp5', 'Actual', 'Predicted', 'Error'])

        # Filter ocean and land grids
        ocean_data = (total_data[total_data['ocean_mask'] == 1])
        land_data = (total_data[total_data['land_mask'] == 1])

        # Get year and month
        mnth = total_data['month']
        year = total_data['year']
        year_month = str(int(year[0])) + '-' + str(int(mnth[0]))

        if monthly:
            # Make directory
            if(os.path.exists(year_month)==False):
                os.mkdir(year_month)
            # Make heatmaps and get land ocean grids
            os.chdir(year_month)
            ocean_grid, land_grid = heatmaps(ocean_data, land_data)
            mean_ocean_error.append(ocean_grid)
            mean_land_error.append(land_grid)

        else:
            # Get overall error of whole test set
            mean_ocean_error = np.array(mean_ocean_error)
            mean_ocean_error = mean_ocean_error.mean(0)

            mean_land_error = np.array(mean_land_error)
            mean_land_error = mean_land_error.mean(0)
            heatmaps(mean_ocean_error, mean_land_error, monthly=False)

        # Plot remaining graphs
        stats(y_val, yhat, y_mean)
        scatterplot(total_data)
        pdf_plot(total_data)

    # Make plots for each month in test set
    for month in tqdm(range(0, len(X_val_total)), desc="Evaluating model"):
        eval_helper(ensemble, X_val_total[month], y_val_total[month], mean_ocean_error, mean_land_error)
        os.chdir('..')

    # Make plot for overall performance
    eval_helper(ensemble, X_val_total, y_val_total, mean_ocean_error, mean_land_error, monthly=False)
