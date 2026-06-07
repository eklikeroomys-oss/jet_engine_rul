# Jet Engine RUL Using NASA CMAPSS Data
# "Data URL: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from enum import Enum
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Enum with column indexes
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

# Feature Engineered Columns:
RUL_COLUMN = "RUL"
RUL_CLIPPED_COLUMN = "RUL_CLIPPED"
CONDITIONS_COLUMN = "Operational Condition"

# Constants from given data
NUM_OPERATIONAL_CONDITIONS = 6
OPERATIONAL_PARAMS = [COLUMN_NAMES[Column.Altitude], COLUMN_NAMES[Column.MachNumber], COLUMN_NAMES[Column.TRA]]

# Parameter Tuning:
WINDOW_SIZE = 10
MIN_SAMPLES_LEAF = 4
TRAINING_FILE_COUNT = 4
RANDOM_STATE=42

def printHeading(heading):
    c = "#"
    length = len(heading)
    print(f"\n{c * (length + 4)}")
    print(f"{c} {heading} {c}")
    print(f"{c * (length + 4)}")

def LoadData(debug=False):
    printHeading("Data Gathering")
    df_train_files = []
    head_count = 5
    for i in range(TRAINING_FILE_COUNT):
        print(f"Loading training file {i+1}...")
        df_train_files.append(pd.read_csv(f"../Data/train_FD00{i+1}.txt", sep=' ', header=None))
        if debug: 
            print(df_train_files[i].head(head_count))
            print(df_train_files[i].describe())

    print("Uniquifying unit numbers...")
    # Unit numbers are duplicated between the training files. 
    # If we can uniquify them, we can merge the data sets into one.
    for i in range(TRAINING_FILE_COUNT):
        if (i > 0):
            df_train_files[i][Column.UnitNumber.value] += \
                df_train_files[i-1][Column.UnitNumber.value].max()

    print("Merging data sets...")
    # The data can now be merged into a single data set as we have unique 
    # Unit Numbers.
    df_train = pd.concat(df_train_files)
    if debug:
        print(df_train)
        print(df_train.describe())
    return df_train

def PrepareData(df_train, debug=False):
    printHeading("Data Preparation")
    print("Removing null data...")
    # The data has columns 26 and 27 which should not be present. 
    # Inspecting these columns, it looks like they are the result of trailing spaces in the data.
    # We can drop columns 26 and 27 from the data sets.
    df_train = df_train.iloc[:, :26]

    print("Adding column names to dataset...")
    # The first two column names are specified in the readme.txt file attached
    # to the CMAPSS data: Unit Number and Time in Cycles.

    # The paper titled "Damage Propagation Modelling" attached to the CMAPSS data specifies the operational parameters as:
    # 1. Altitude (0-42K ft.)
    # 2. Mach number (0-0.84)
    # 3. Throttle resolver angle (TRA) (20-100)

    # Looking at the min and max values in the dataframe description above, we see 
    # that these parameters map roughly to columns 2, 3 and 4, and in the same 
    # order.

    # For now we will assume that the sensor data is given in the same order as the 
    # specified in "Damage Propagation Modelling".
    df_train.columns = list(COLUMN_NAMES.values())
    print(list(df_train.columns))

    if debug:
        print(df_train.describe())
    return df_train

def FeatureEngineering(df_train, debug=False):
    printHeading("Feature Engineering")
    def CalculateRUL(df_train, debug=False):
        print("Calculating the RUL for each Unit...")
        fail_times = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[COLUMN_NAMES[Column.TimeCycles]].transform('max')
        df_train[RUL_COLUMN] = fail_times - df_train[COLUMN_NAMES[Column.TimeCycles]]

        print("Plotting the RUL distribution for each Unit...")
        max_cycles_per_unit = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[RUL_COLUMN].max()
        plt.hist(max_cycles_per_unit, bins=30)
        plt.title('Distribution of Engine Lifespans')
        plt.xlabel("Engine Lifespan")
        plt.ylabel("Engine Count")
        plt.savefig(f"{OUTPUT_DIR}/Engine_Lifespan_Distribution.png", bbox_inches='tight', dpi=300)
        plt.close()

        if debug:
            print(max_cycles_per_unit.describe())

        # We see from the plot above that most engines fail between ~150 and 280 cycles. The most common lifespan is around 200 cycles. 
        # Very few engines last 400-550 cycles.
        # Engines start failing at about 128 cycles, so we should be able to clip RUL <= 130 cycles.
        return df_train

    def ClusterOperationalConditions(df_train, debug=False):
        print("\nClustering the operational conditions...")
        df_operational_params = df_train[OPERATIONAL_PARAMS]

        print("\tScaling condition data before KMeans fit")
        operational_params_scaler = StandardScaler()
        df_operational_params_scaled = operational_params_scaler.fit_transform(df_operational_params)

        print("\tPerforming KMeans fit")
        km = KMeans(n_clusters=NUM_OPERATIONAL_CONDITIONS)
        df_train[CONDITIONS_COLUMN] = km.fit_predict(df_operational_params_scaled)
        print("\t\tCluster centers :")
        operational_condition_centers = operational_params_scaler.inverse_transform(km.cluster_centers_)
        for i in range(NUM_OPERATIONAL_CONDITIONS):
            print(f"\t\tCondition {i+1}: {OPERATIONAL_PARAMS[0]} = {int(operational_condition_centers[i][0] * 1000)}Ft, \
                    {OPERATIONAL_PARAMS[1]} = {round(operational_condition_centers[i][1], 2)}, {OPERATIONAL_PARAMS[2]} = {round(operational_condition_centers[i][2], 2)}")

        print("\tCalculating operating condition clusters silhouette score...")
        print(f"\t\tResult: {silhouette_score(df_operational_params_scaled, df_train[CONDITIONS_COLUMN], metric="euclidean", sample_size=5000, random_state=RANDOM_STATE)}")

        print("\tPlotting clusters...")
        operational_conditions_fig = plt.figure(figsize=(10,8))
        ax = operational_conditions_fig.add_subplot(111, projection='3d')
        operational_conditions_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        for cluster in range(NUM_OPERATIONAL_CONDITIONS):
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
        plt.savefig(f"{OUTPUT_DIR}/Operational_Condition_Clusters.png", bbox_inches='tight', dpi=300)
        plt.close()

        print("Dropping operational parameter columns, we are satisfied with the clustering...")
        df_train = df_train.drop(columns=OPERATIONAL_PARAMS)

        print("Plotting operational condition over cycles for a few sample engines...")
        sample_units = random.sample(list(df_train[COLUMN_NAMES[Column.UnitNumber]].unique()), 10)
        plt.figure(figsize=(12, 8))
        for unit in sample_units:
            unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
            unit_data = unit_data.sort_values(COLUMN_NAMES[Column.TimeCycles])
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], 
                    unit_data[CONDITIONS_COLUMN], 
                    label=f'Unit {unit}')
        
        plt.xlabel('Cycle')
        plt.ylabel('Operational Condition')
        plt.title('Operational Condition over Time for Sample Engines')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/Operational_Condition_Per_Engine_Sample.png", bbox_inches='tight', dpi=300)
        plt.close()
        # From the graph above it is clear that each engine operates at many different operating conditions, not just one.
        # We should try to calculate an average of all previous conditions at each time step for the final result.

        if debug:
            print(df_train)

        return df_train

    def ClipRUL(df_train, debug=False):
        rul_limit = 130
        print(f"\nClipping RUL at a maximum of {rul_limit} cycles...")
        df_train[RUL_CLIPPED_COLUMN] = df_train[RUL_COLUMN].clip(upper=rul_limit)

        print("Plotting the clipped RUL distribution for each Unit...")
        max_cycles_per_unit = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[RUL_CLIPPED_COLUMN].max()
        plt.hist(max_cycles_per_unit, bins=5)
        plt.title('Distribution of Clipped Engine Lifespans')
        plt.xlabel("Engine Lifespan")
        plt.ylabel("Engine Count")
        plt.savefig(f"{OUTPUT_DIR}/Engine_Lifespan_Clipped_Distribution.png", bbox_inches='tight', dpi=300)
        plt.close()

        if debug:
            print(df_train.describe())

        return df_train

    def FeatureSelection(df_train, debug=False):
        ## Feature Selection
        ###         Find and drop sensors that have low variance for all operational conditions:
        sensor_columns = [COLUMN_NAMES[col] for col in Column if Column.T2.value <= col.value <= Column.W32.value]
        low_variance_sensors = []
        sensor_variance_limit = 0.01
        for i in range(NUM_OPERATIONAL_CONDITIONS):
            sensor_variances = df_train[df_train[CONDITIONS_COLUMN] == i][sensor_columns].var()
            low_variance_sensors.append(sensor_variances[sensor_variances < sensor_variance_limit].index.tolist())
            if debug:
                print(f"\nCondition {i} Sensor Variances:\n{sensor_variances}")
                print(f"\nCondition {i} Low Variance Sensors:\n{low_variance_sensors[i]}")

        common_low_variance_sensors = set(low_variance_sensors[0]).intersection(*low_variance_sensors[1:])
        df_train = df_train.drop(columns=list(common_low_variance_sensors))
        sensor_columns = list(set(sensor_columns) - set(common_low_variance_sensors))
        if debug:
            print(common_low_variance_sensors)
            print(df_train.describe())

        return df_train, sensor_columns

    def CreateRollingFeatures(df_train, sensor_columns, debug=False):
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


        def add_rolling_features(group):
            for c in sensor_columns:
                group[f"{c}_ROLL_MEAN"] = group[c].rolling(window=WINDOW_SIZE, min_periods=1).mean()
                group[f"{c}_ROLL_STD"] = group[c].rolling(window=WINDOW_SIZE, min_periods=1).std()
                # group[f"{c}_ROLL_MIN"] = group[c].rolling(window=window_size, min_periods=1).min()  # Min/max features overdominant, resulting in overfitting, removed.
                # group[f"{c}_ROLL_MAX"] = group[c].rolling(window=window_size, min_periods=1).max()  # Min/max features overdominant, resulting in overfitting, removed.

                group[f'{c}_DELTA'] = group[c].diff(periods=1) # change from previous cycle
                group[f'{c}_ROLL_SLOPE'] = group[c].diff(WINDOW_SIZE) / WINDOW_SIZE
            return group

        df_train = df_train.groupby(COLUMN_NAMES[Column.UnitNumber]).apply(add_rolling_features).reset_index()
        df_train = df_train.drop(columns=['level_1'])
        df_train = df_train.bfill()

        return df_train

    df_train = CalculateRUL(df_train)
    df_train = ClipRUL(df_train)
    df_train = ClusterOperationalConditions(df_train)
    #df_train, sensor_columns = FeatureSelection(df_train)
    #df_train = CreateRollingFeatures(df_train, sensor_columns)
    sensor_columns = []
    return df_train, sensor_columns

def TrainTestSplit(df_train, debug=False):
    ## Train/Test Split
    unique_units = df_train[COLUMN_NAMES[Column.UnitNumber]].unique()
    train_units, test_units = train_test_split(unique_units, test_size=0.20, random_state=RANDOM_STATE)
    df_train_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(train_units)].copy()
    df_test_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(test_units)].copy()
    if debug:
        print(f"Train engines: {len(train_units)}, Test engines: {len(test_units)}")
    return df_train_split, df_test_split

def RandomForestModel(df_train_split, df_test_split, debug=False):
    # Training the Random Forest Model:
    ## Extract X and Y data:
    target_col = RUL_CLIPPED_COLUMN
    exclude_cols = [
            COLUMN_NAMES[Column.UnitNumber], 
            COLUMN_NAMES[Column.TimeCycles], 
            target_col, 
            RUL_COLUMN
            ]
    #exclude_cols.extend(sensor_columns) # We only want the new features we added.
    feature_cols = [col for col in df_train_split.columns 
                    if col not in exclude_cols
                    and ('ROLL' in col or 'DELTA' in col or 'CONDITION' in col)]
    strong_raw = ['NRf', 'Ps30', 'Nc', 'T50', 'phi']
    feature_cols.extend(strong_raw)

    if debug:
        print(f"Number of features: {len(feature_cols)}")

    X_train = df_train_split[feature_cols]
    y_train = df_train_split[target_col]

    X_test = df_test_split[feature_cols]
    y_test = df_test_split[target_col]

    ## Train the model:
    # Train
    model = RandomForestRegressor(
            n_estimators=400,     # number of trees
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_depth=None,       # limit depth to prevent overfitting
            n_jobs=-1,            # use all CPU cores
    )

    model.fit(X_train, y_train)
    if debug:
        print("✅ Model trained!")

        feature_importances = pd.Series(model.feature_importances_, index=feature_cols)
        print(feature_importances.sort_values(ascending=False).head(20))

    return model, feature_cols, X_test, y_test, X_train, y_train

def EvaluateModel(df_test_split, model, feature_cols, X_test, y_test, debug=True):
    ## Evaluate the model:
    y_pred = model.predict(X_test)

    ## Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")

    # NASA's score (common for RUL)
    def rul_score(y_test, y_pred):
        diff = y_pred - y_test
        score = np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))
        return score
    print(f"NASA Score: {rul_score(y_test, y_pred):.2f}")

    def plot_ytest_vs_ypred():
        # Scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, s=10)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Perfect prediction line

        plt.xlabel('Actual RUL (Clipped)')
        plt.ylabel('Predicted RUL')
        plt.title('Actual vs Predicted RUL')
        plt.grid(True)
        plt.show()
    plot_ytest_vs_ypred()

    def plot_predictions_random_engines(df_test_split):
        sample_units = random.sample(list(df_test_split[COLUMN_NAMES[Column.UnitNumber]].unique()), 5)

        plt.figure(figsize=(12, 8))

        for unit in sample_units:
            unit_data = df_test_split[df_test_split[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
            unit_data = unit_data.sort_values(COLUMN_NAMES[Column.TimeCycles])
            
            actual = unit_data[RUL_CLIPPED_COLUMN]
            pred = model.predict(unit_data[feature_cols])
            
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], actual, label=f'Unit {unit} - Actual', linestyle='-', marker='o')
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], pred, label=f'Unit {unit} - Predicted', linestyle='--')

        plt.xlabel('Cycle')
        plt.ylabel('RUL')
        plt.title('RUL Prediction over Cycles for Sample Engines')
        plt.legend()
        plt.grid(True)
        plt.show()
    plot_predictions_random_engines(df_test_split)



df_train = LoadData()
df_train = PrepareData(df_train)
df_train, sensor_columns = FeatureEngineering(df_train)

#df_train = df_train[df_train[CONDITIONS_COLUMN] == 0]
#df_train_split, df_test_split = TrainTestSplit(df_train)
#model, feature_cols, X_test, y_test, X_train, y_train = RandomForestModel(df_train_split, df_test_split)

#EvaluateModel(df_train_split, model, feature_cols, X_train, y_train)
#EvaluateModel(df_test_split, model, feature_cols, X_test, y_test)
