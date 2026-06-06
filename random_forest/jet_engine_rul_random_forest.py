# Jet Engine RUL Using NASA CMAPSS Data
# "Data URL: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enum import Enum
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

debug = False
def debugPrint(msg):
    if debug:
        print(msg)

class Column(Enum):
    UnitNumber = 0
    TimeCycles = 1
    Altitude = 2
    MachNumber = 3
    TRA = 4
    T2 = 5
    T24 = 6
    T30 = 7
    T50 = 8
    P2 = 9
    P15 = 10
    P30 = 11
    Nf = 12
    Nc = 13
    epr = 14
    Ps30 = 15
    phi = 16
    NRf = 17
    NRc = 18
    BPR = 19
    farB = 20
    htBleed = 21
    Nf_dmd = 22
    PCNfR_dmd = 23
    W31 = 24
    W32 = 25

# Dictionary mapping Column enum values to their string names
COLUMN_NAMES = {
    Column.UnitNumber: 'Unit Number',
    Column.TimeCycles: 'Time (Cycles)',
    Column.Altitude: 'Altitude',
    Column.MachNumber: 'Mach Number',
    Column.TRA: 'TRA',
    Column.T2: 'T2',  # Total temperature at fan inlet (°R)
    Column.T24: 'T24',  # Total temperature at LPC outlet (°R)
    Column.T30: 'T30',  # Total temperature at HPC outlet (°R)
    Column.T50: 'T50',  # Total temperature at LPT outlet (°R)
    Column.P2: 'P2',  # Pressure at fan inlet (psia)
    Column.P15: 'P15',  # Total pressure in bypass-duct (psia)
    Column.P30: 'P30',  # Total pressure at HPC outlet (psia)
    Column.Nf: 'Nf',  # Physical fan speed (rpm)
    Column.Nc: 'Nc',  # Physical core speed (rpm)
    Column.epr: 'epr',  # Engine pressure ratio (P50/P2) (--)
    Column.Ps30: 'Ps30',  # Static pressure at HPC outlet (psia)
    Column.phi: 'phi',  # Ratio of fuel flow to Ps30 (pps/psi)
    Column.NRf: 'NRf',  # Corrected fan speed (rpm)
    Column.NRc: 'NRc',  # Corrected core speed (rpm)
    Column.BPR: 'BPR',  # Bypass Ratio (--)
    Column.farB: 'farB',  # Burner fuel-air ratio (--)
    Column.htBleed: 'htBleed',  # Bleed Enthalpy (--)
    Column.Nf_dmd: 'Nf_dmd',  # Demanded fan speed (rpm)
    Column.PCNfR_dmd: 'PCNfR_dmd',  # Demanded corrected fan speed (rpm)
    Column.W31: 'W31',  # HPT coolant bleed (lbm/s)
    Column.W32: 'W32',  # LPT coolant bleed (lbm/s)
}

# Data Gathering
## Data Loading
###     Start by loading the training data.
training_file_count = 4
df_train_files = []
for i in range(training_file_count):
    df_train_files.append(pd.read_csv(f"../Data/train_FD00{i+1}.txt", sep=' ', header=None))
    debugPrint(f"\nTraining File {i+1}")
    debugPrint(df_train_files[i].describe())

## Uniquifying Unit Numbers
###     Unit numbers are duplicated between the training files. 
###     If we can uniquify them, we can merge the data sets into one.
for i in range(training_file_count):
    if (i > 0):
        df_train_files[i][Column.UnitNumber.value] += \
            df_train_files[i-1][Column.UnitNumber.value].max()

## Merging Data Sets
###     The data can now be merged into a single data set as we have unique 
###     Unit Numbers.
df_train = pd.concat(df_train_files)

# Data Preparation
## Removing null data
###     The data has columns 26 and 27 which should not be present. 
###     Inspecting these columns, it looks like they are the result of trailing 
###     spaces in the data.
###     We can drop columns 26 and 27 from the data sets.
df_train = df_train.iloc[:, :26]

## Defining Columns
###     The first two column names are specified in the readme.txt file attached
###     to the CMAPSS data: Unit Number and Time in Cycles.
###     
###     The paper titled "Damage Propagation Modelling" attached to the CMAPSS data 
###     specifies the operational parameters as:
###         1. Altitude (0-42K ft.)
###         2. Mach number (0-0.84)
###         3. Throttle resolver angle (TRA) (20-100)
###  
###     Looking at the min and max values in the dataframe description above, we see 
###     that these parameters map roughly to columns 2, 3 and 4, and in the same 
###     order.
###
###     For now we will assume that the sensor data is given in the same order as the 
###     specified in "Damage Propagation Modelling".
df_train.columns = list(COLUMN_NAMES.values())

###     Have a look at the description of the complete training set:
debugPrint("\nComplete training set description:")
debugPrint(df_train.describe())

# Feature Engineering
RUL_COLUMN = "RUL"
## Compute the RUL for each Unit
fail_times = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[COLUMN_NAMES[Column.TimeCycles]].transform('max')
df_train[RUL_COLUMN] = fail_times - df_train[COLUMN_NAMES[Column.TimeCycles]]

## Clustering the operational conditions
###     Data is given for six operational conditions determined by the Altitude, 
###     Mach Number and TRA columns.
operational_params = [
        COLUMN_NAMES[Column.Altitude],
        COLUMN_NAMES[Column.MachNumber],
        COLUMN_NAMES[Column.TRA]
        ]
df_operational_params = df_train[operational_params]

### Scale the condition data before KMeans fit
operational_params_scaler = StandardScaler()
df_operational_params_scaled = operational_params_scaler.fit_transform(df_operational_params)

### Perform KMeans fit
CONDITIONS_COLUMN = "Operational Condition"
num_operational_conditions = 6 # From "Damage Propagation Modelling"
km = KMeans(n_clusters=num_operational_conditions)
df_train[CONDITIONS_COLUMN] = km.fit_predict(df_operational_params_scaled)
debugPrint(df_train)

if debug:
    operational_conditions_fig = plt.figure(figsize=(10,8))
    ax = operational_conditions_fig.add_subplot(111, projection='3d')
    operational_conditions_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
    for cluster in range(num_operational_conditions):
        cluster_data = df_train[df_train[CONDITIONS_COLUMN] == cluster]

        ax.scatter(
            cluster_data[COLUMN_NAMES[Column.Altitude]],
            cluster_data[COLUMN_NAMES[Column.MachNumber]],
            cluster_data[COLUMN_NAMES[Column.TRA]],
            c=operational_conditions_colors[cluster],
            label=f'Condition {cluster}',
            s=50,
            alpha=0.8
        )
    ax.set_xlabel(f"{COLUMN_NAMES[Column.Altitude]} (x1000 ft)")
    ax.set_ylabel(COLUMN_NAMES[Column.MachNumber])
    ax.set_zlabel(COLUMN_NAMES[Column.TRA])
    ax.set_title("Operational Condition Clusters")
    ax.legend()
    plt.show()

debugPrint("\nOperation Condition Clusters Silhouette Score:")
debugPrint(silhouette_score(df_operational_params_scaled, df_train[CONDITIONS_COLUMN], metric="euclidean", sample_size=25000))

debugPrint("Cluster centers :")
operational_condition_centers = operational_params_scaler.inverse_transform(km.cluster_centers_)
for i in range(num_operational_conditions):
    debugPrint(f"Condition {i+1}: {operational_params[0]} = {int(operational_condition_centers[i][0] * 1000)}Ft, \
            {operational_params[1]} = {round(operational_condition_centers[i][1], 2)}, {operational_params[2]} = {round(operational_condition_centers[i][2], 2)}")

### Drop operational parameter columns, we are satisfied with the clustering:
df_train = df_train.drop(columns=operational_params)
debugPrint(df_train)

## Limiting the dataset to a lower RUL.
max_cycles_per_unit = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[COLUMN_NAMES[Column.TimeCycles]].max()
debugPrint(f"\nMax cycles per unit: \n{max_cycles_per_unit}")
if debug:
    plt.hist(max_cycles_per_unit, bins=30)
    plt.title('Distribution of Engine Lifespans')
    plt.xlabel("Engine Lifespan")
    plt.ylabel("Engine Count")
    plt.show()
debugPrint(max_cycles_per_unit.describe())

###         We see from the plot below that most engines fail between ~150 and 280 cycles. The most common lifespan is around 200 cycles. 
###         Very few engines last 400-550 cycles.
###         Engines start failing at about 128 cycles, so we should be able to limit our dataset to RUL <= 130 cycles.
rul_limit = 130
df_train = df_train[df_train[RUL_COLUMN] <= rul_limit]
debugPrint(df_train.describe())

## Feature Selection
###         Find and drop sensors that have low variance for all operational conditions:
sensor_columns = [COLUMN_NAMES[col] for col in Column if Column.T2.value <= col.value <= Column.W32.value]
low_variance_sensors = []
sensor_variance_limit = 0.01
for i in range(num_operational_conditions):
    sensor_variances = df_train[df_train[CONDITIONS_COLUMN] == i][sensor_columns].var()
    debugPrint(f"\nCondition {i} Sensor Variances:\n{sensor_variances}")
    low_variance_sensors.append(sensor_variances[sensor_variances < sensor_variance_limit].index.tolist())
    debugPrint(f"\nCondition {i} Low Variance Sensors:\n{low_variance_sensors[i]}")

common_low_variance_sensors = set(low_variance_sensors[0]).intersection(*low_variance_sensors[1:])
debugPrint(common_low_variance_sensors)
df_train = df_train.drop(columns=list(common_low_variance_sensors))
sensor_columns = list(set(sensor_columns) - set(common_low_variance_sensors))
debugPrint(df_train.describe())

## Create Rolling Features
###         Sort and re-index training data.
df_train = df_train.sort_values([COLUMN_NAMES[Column.UnitNumber], COLUMN_NAMES[Column.TimeCycles]]).reset_index(drop=True)

###         Calculate rolling features
###         Random Forest does not understand time or sequences by itself.
###         It looks at one row at a time and makes a prediction based only on the numbers in that row.
###
###         Having only the current sensor readings (e.g. temperature, pressure at cycle 150), the model 
###         has no idea whether those values are:
###          * Normal (early in the engine’s life), or
###          * Getting worse (late in life, close to failure).
###         It cannot see the trend or history.
###
###         Rolling features give the model:
###          * The average value over the last X cycles
###          * Quantifies the recent change in a value
###          * Stability of a value
###
###         This gives the model context about degradation, which is the key signal for predicting 
###         Remaining Useful Life (RUL).
###
###         Without these, Random Forest will perform quite poorly. With them, it becomes much 
###         smarter at detecting when an engine is starting to fail.

df_train_orig = df_train.copy()
for window_size in [10, 20, 30, 40, 50]:
    df_train = df_train_orig.copy()
    print(f"window_size={window_size}")

    def add_rolling_features(group):
        for c in sensor_columns:
            group[f"{c}_ROLL_MEAN"] = group[c].rolling(window=window_size, min_periods=1).mean()
            group[f"{c}_ROLL_STD"] = group[c].rolling(window=window_size, min_periods=1).std()
            group[f"{c}_ROLL_MIN"] = group[c].rolling(window=window_size, min_periods=1).min()
            group[f"{c}_ROLL_MAX"] = group[c].rolling(window=window_size, min_periods=1).max()

            group[f'{c}_ROLL_DELTA'] = group[c].diff() # change from previous cycle
            group[f'{c}_ROLL_SLOPE'] = group[c].diff(window_size) / window_size # rough slope
        return group

    df_train = df_train.groupby(COLUMN_NAMES[Column.UnitNumber]).apply(add_rolling_features).reset_index()
    df_train = df_train.drop(columns=['level_1'])
    df_train = df_train.bfill()

    ## Train/Test Split
    unique_units = df_train[COLUMN_NAMES[Column.UnitNumber]].unique()
    train_units, test_units = train_test_split(unique_units, test_size=0.20)
    df_train_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(train_units)].copy()
    df_test_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(test_units)].copy()
    print(f"Train engines: {len(train_units)}, Test engines: {len(test_units)}")

    # Training the Random Forest Model:
    ## Extract X and Y data:
    target_col = RUL_COLUMN
    exclude_cols = [COLUMN_NAMES[Column.UnitNumber], COLUMN_NAMES[Column.TimeCycles], target_col]   # adjust if needed
    feature_cols = [col for col in df_train_split.columns if col not in exclude_cols]
    print(f"Number of features: {len(feature_cols)}")

    X_train = df_train_split[feature_cols]
    y_train = df_train_split[target_col]

    X_test = df_test_split[feature_cols]
    y_test = df_test_split[target_col]

    ## Train the model:
    # Train
    model = RandomForestRegressor(
            n_estimators=200,     # number of trees
            max_depth=20,         # limit depth to prevent overfitting
            n_jobs=-1,            # use all CPU cores
    )

    model.fit(X_train, y_train)
    print("✅ Model trained!")

    ## Evaluate the model:
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")

    # NASA's score (common for RUL)
    def rul_score(y_true, y_pred):
        diff = y_pred - y_true
        score = np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))
        return score

    print(f"NASA Score: {rul_score(y_test, y_pred):.2f}")

    feature_importances = pd.Series(model.feature_importances_, index=feature_cols)
    print(feature_importances.sort_values(ascending=False).head(20))
