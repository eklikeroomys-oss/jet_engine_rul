# Jet Engine RUL Using NASA CMAPSS Data
# "Data URL: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import argparse

from enum import Enum
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, silhouette_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


class Set(Enum):
    FD001 = 1
    FD002 = 2
    FD003 = 3
    FD004 = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jet Engine RUL Prediction using Random Forest"
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        choices=['FD001', 'FD002', 'FD003', 'FD004'],
        default='FD001',
        help='Dataset to use (default: FD001)'
    )
    parser.add_argument(
        '-t', '--tune',
        action='store_true',
        help='Enable hyperparameter tuning with GridSearchCV'
    )
    args = parser.parse_args()
    return Set[args.dataset], args.tune


CURRENT_SET, TUNE = parse_args()

if CURRENT_SET == Set.FD001:
    EMA_SPAN = 15 # Best window for smoothing
    MEAN_WINDOW = 5 # Best window for calculating mean
    STD_WINDOW = 100 # Best window for calculating STD
    # Parameter tuning using GridSearchCV:
    NUM_TREES = 50
    MAX_DEPTH = 12
    MIN_SAMPLES_LEAF = 2
    MIN_SAMPLES_SPLIT = 18
    MAX_FEATURES = 'sqrt'
    # Final training results FD001:
    # Best Estimator: RandomForestRegressor(max_depth=12, max_features='sqrt', min_samples_leaf=2, min_samples_split=18, n_estimators=50, n_jobs=-1, random_state=42)
    # Validation using train/test split:
    #   RMSE (Training Split): 3.26
    #   NASA Score (Training Split): 3847
    #   RMSE (Testing Split): 11.78
    #   NASA Score (Testing Split): 13076
    pass
elif CURRENT_SET == Set.FD002:
    EMA_SPAN = 25 # Best window for smoothing
    MEAN_WINDOW = 5 # Best window for calculating mean
    STD_WINDOW = 75 # Best window for calculating STD
    # Parameter tuning using GridSearch:
    NUM_TREES = 390
    MAX_DEPTH = 18
    MIN_SAMPLES_LEAF = 2
    MIN_SAMPLES_SPLIT = 6
    MAX_FEATURES = 'sqrt'
    # Final training results FD002:
    # Best Estimator: RandomForestRegressor(max_depth=18, max_features='sqrt', min_samples_leaf=2, min_samples_split=6, n_estimators=390, n_jobs=-1, random_state=42)
    # Validation using train/test split:
    #   RMSE (Training Split): 1.19
    #   NASA Score (Training Split): 2986
    #   RMSE (Testing Split): 12.62
    #   NASA Score (Testing Split): 49015
    pass
elif CURRENT_SET == Set.FD003:
    EMA_SPAN = 10 # Best window for smoothing
    MEAN_WINDOW = 25 # Best window for calculating mean
    STD_WINDOW = 55 # Best window for calculating STD
    # Parameter tuning using GridSearch:
    NUM_TREES = 430
    MAX_DEPTH = 16
    MIN_SAMPLES_LEAF = 2
    MIN_SAMPLES_SPLIT = 2
    MAX_FEATURES = 'sqrt'
    # Final training results FD003:
    # Best Estimator: RandomForestRegressor(max_depth=16, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, n_estimators=430, n_jobs=-1, random_state=42)
    # Validation using train/test split:
    #   RMSE (Training Split): 1.10
    #   NASA Score (Training Split): 1173
    #   RMSE (Testing Split): 9.56
    #   NASA Score (Testing Split): 11637
    pass
elif CURRENT_SET == Set.FD004:
    EMA_SPAN = 30 # Best window for smoothing
    MEAN_WINDOW = 15 # Best window for calculating mean
    STD_WINDOW = 65 # Best window for calculating STD
    # Parameter Tuning From GridSearch:
    NUM_TREES = 450
    MAX_DEPTH = 10
    MIN_SAMPLES_LEAF = 4
    MIN_SAMPLES_SPLIT = 14
    MAX_FEATURES = 'sqrt'
    # Final training results FD004:
    # Best Estimator: RandomForestRegressor(max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=14, n_estimators=450, n_jobs=-1, random_state=42)
    # Validation using train/test split:
    #   RMSE (Training Split): 7.10
    #   NASA Score (Training Split): 54620
    #   RMSE (Testing Split): 14.80
    #   NASA Score (Testing Split): 96428
    pass

NASA_SAFETY_BUFFER = 0
RUL_LIMIT = 100

RANDOM_STATE = 42
PARAM_GRIDS = {
    Set.FD001: {
        'n_estimators': [50],
        'max_depth': [12],
        'min_samples_leaf': [2],
        'min_samples_split': range(2, 20, 2),
        'max_features': ['sqrt'],
        'random_state': [RANDOM_STATE],
        'n_jobs': [-1]
    },
    Set.FD002: {
        'n_estimators': [390],
        'max_depth': [18],
        'min_samples_leaf': [2],
        'min_samples_split': range(2, 20, 2),
        'max_features': ['sqrt'],
        'random_state': [RANDOM_STATE],
        'n_jobs': [-1]
    },
    Set.FD003: {
        'n_estimators': [430],
        'max_depth': [16],
        'min_samples_leaf': [2],
        'min_samples_split': range(2, 20, 2),
        'max_features': ['sqrt'],
        'random_state': [RANDOM_STATE],
        'n_jobs': [-1]
    },
    Set.FD004: {
        'n_estimators': [450],
        'max_depth': [10],
        'min_samples_leaf': [4],
        'min_samples_split': range(2, 20, 2),
        'max_features': ['sqrt'],
        'random_state': [RANDOM_STATE],
        'n_jobs': [-1]
    },
}


training_files = {
        Set.FD001: "../Data/train_FD001.txt",
        Set.FD002: "../Data/train_FD002.txt",
        Set.FD003: "../Data/train_FD003.txt",
        Set.FD004: "../Data/train_FD004.txt",
        }

testing_input_files = {
        Set.FD001: "../Data/test_FD001.txt",
        Set.FD002: "../Data/test_FD002.txt",
        Set.FD003: "../Data/test_FD003.txt",
        Set.FD004: "../Data/test_FD004.txt",
        }

testing_RUL_files = {
        Set.FD001: "../Data/RUL_FD001.txt",
        Set.FD002: "../Data/RUL_FD002.txt",
        Set.FD003: "../Data/RUL_FD003.txt",
        Set.FD004: "../Data/RUL_FD004.txt",
        }

OUTPUT_DIR = Path(f"output/{CURRENT_SET.name}")

RUL_DIR = Path(f"{OUTPUT_DIR}/1_RUL")
CONDITION_DIR = Path(f"{OUTPUT_DIR}/2_Conditions")
SENSORS_DIR = Path(f"{OUTPUT_DIR}/3_Sensors")
RESULTS_DIR = Path(f"{OUTPUT_DIR}/4_Results")

RUL_DIR.mkdir(parents=True, exist_ok=True)
CONDITION_DIR.mkdir(parents=True, exist_ok=True)
SENSORS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

RUL_COLUMN = "RUL"
RUL_CLIPPED_COLUMN = "RUL_CLIPPED"
CONDITIONS_COLUMN = "Operational Condition"

# Constants from given data
NUM_OPERATIONAL_CONDITIONS = {
        Set.FD001: 1,
        Set.FD002: 6,
        Set.FD003: 1,
        Set.FD004: 6
}

OPERATIONAL_PARAMS = [COLUMN_NAMES[Column.Altitude], COLUMN_NAMES[Column.MachNumber], COLUMN_NAMES[Column.TRA]]
SILHOUETTE_SCORE_SAMPLE_SIZE = 5000 # Silhouette score sample size for operational condition clustering.
SAMPLE_UNITS = 25
TEST_SIZE = 0.3 # Train/test split size

DPI=100
HEAD_COUNT = 5

def TrainTestDescription(training: bool):
    return "Training Data" if training else "Testing Data"

def printHeading(heading, training: bool):
    c = "#"
    heading_complete = f"{heading} ({TrainTestDescription(training)})"
    length = len(heading_complete)
    print(f"\n{c * (length + 4)}")
    print(f"{c} {heading_complete} {c}")
    print(f"{c * (length + 4)}")

def printBusy(msg, training: bool):
    msg_complete = f"{msg} ({TrainTestDescription(training)})..."
    print(f"{msg_complete}")

def plotTitle(title, training: bool):
    return f"{TrainTestDescription(training)} {title}"

def LoadData(training: bool, debug=False):
    printHeading("Data Gathering", training)
    printBusy(f"Loading data files for {CURRENT_SET.name}", training)
    df = pd.read_csv(training_files[CURRENT_SET] if training else testing_input_files[CURRENT_SET], sep=' ', header=None)

    if debug: 
        print(df.head(HEAD_COUNT))
        print(df.describe())

    return df

def PlotHistogram(data, bins, dir, title, xlabel, ylabel):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{dir}/{title.replace(" ", "_")}.png", bbox_inches='tight', dpi=DPI)
    plt.close()

def PrepareData(df, training: bool, debug=False):
    printHeading(f"Data Preparation", training)
    printBusy("Removing null data", training)
    # The data has columns 26 and 27 which should not be present. 
    # Inspecting these columns, it looks like they are the result of trailing spaces in the data.
    # We can drop columns 26 and 27 from the data sets.
    df = df.iloc[:, :26]

    printBusy("Adding column names to dataset", training)
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
    df.columns = list(COLUMN_NAMES.values())
    print(list(df.columns))

    printBusy("Sorting data by unit number and time (cycles)", training)
    # Ensure data is sorted
    df = df.sort_values([COLUMN_NAMES[Column.UnitNumber], 
                                   COLUMN_NAMES[Column.TimeCycles]]).reset_index(drop=True)
    if debug:
        print(df)
        print(df.describe())
    return df

def FeatureEngineering(df, training: bool, debug=False):
    printHeading("Feature Engineering", training)

    def CalculateRUL(df_train, debug=False):
        print("Calculating the RUL for each Unit...")
        fail_times = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[COLUMN_NAMES[Column.TimeCycles]].transform('max')
        df_train[RUL_COLUMN] = fail_times - df_train[COLUMN_NAMES[Column.TimeCycles]]

        print("Plotting the RUL distribution for each Unit...")
        max_cycles_per_unit = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[RUL_COLUMN].max()
        PlotHistogram(max_cycles_per_unit, 30, RUL_DIR, 'Distribution of Engine Lifespans', "Engine Lifespan", "Engine Count")

        if debug:
            print(max_cycles_per_unit.describe())

        # We see from the plot above that most engines fail between ~150 and 280 cycles. The most common lifespan is around 200 cycles. 
        # Very few engines last 400-550 cycles.
        # Engines start failing at about 128 cycles, so we should be able to clip RUL <= RUL_LIMIT cycles.
        return df_train

    def ClipRUL(df_train, debug=False):
        rul_limit = RUL_LIMIT
        print(f"\nClipping RUL at a maximum of {rul_limit} cycles...")
        df_train[RUL_CLIPPED_COLUMN] = df_train[RUL_COLUMN].clip(upper=rul_limit)

        print("Plotting the clipped RUL distribution for each Unit...")
        max_cycles_per_unit = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[RUL_CLIPPED_COLUMN].max()
        PlotHistogram(max_cycles_per_unit, 5, RUL_DIR, 'Distribution of Clipped Engine Lifespans', "Engine Lifespan", "Engine Count")

        if debug:
            print(df_train.describe())

        return df_train

    def ClusterOperationalConditions(df, debug=False):
        printBusy("\nClustering operational conditions", training)
        df_operational_params = df[OPERATIONAL_PARAMS].copy()

        printBusy("\tScaling condition data before KMeans fit", training)
        operational_params_scaler = StandardScaler()
        df_operational_params_scaled = operational_params_scaler.fit_transform(df_operational_params)

        printBusy("\tPerforming KMeans fit", training)
        km = KMeans(n_clusters=NUM_OPERATIONAL_CONDITIONS[CURRENT_SET], random_state=RANDOM_STATE)
        df[CONDITIONS_COLUMN] = km.fit_predict(df_operational_params_scaled)
        printBusy(f"\t\tCluster centers", training)
        operational_condition_centers = operational_params_scaler.inverse_transform(km.cluster_centers_)
        for i in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET]):
            print(f"\t\tCondition {i+1}: {OPERATIONAL_PARAMS[0]} = {int(operational_condition_centers[i][0] * 1000)}Ft, \
                    {OPERATIONAL_PARAMS[1]} = {round(operational_condition_centers[i][1], 2)}, {OPERATIONAL_PARAMS[2]} = {round(operational_condition_centers[i][2], 2)}")

        if NUM_OPERATIONAL_CONDITIONS[CURRENT_SET] > 1:
            printBusy(f"\tCalculating operating condition clusters silhouette score...", training)
            print(f"\t\tResult: {silhouette_score(df_operational_params_scaled, df[CONDITIONS_COLUMN], metric="euclidean", sample_size=SILHOUETTE_SCORE_SAMPLE_SIZE, random_state=RANDOM_STATE)}")

        printBusy("\tPlotting clusters...", training)
        operational_conditions_fig = plt.figure(figsize=(10,8))
        ax = operational_conditions_fig.add_subplot(111, projection='3d')
        operational_conditions_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        for cluster in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET]):
            cluster_data = df[df[CONDITIONS_COLUMN] == cluster]

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
        title = plotTitle("Operational Condition Clusters", training)
        ax.set_title(title)
        ax.legend()
        plt.savefig(f"{CONDITION_DIR}/{title.replace(" ", "_")}.png", bbox_inches='tight', dpi=DPI)
        plt.close()

        printBusy("\tPlotting operational condition vs time", training)
        sample_units = random.sample(list(df[COLUMN_NAMES[Column.UnitNumber]].unique()), SAMPLE_UNITS)
        plt.figure(figsize=(12, 8))
        for unit in sample_units:
            unit_data = df[df[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], 
                     unit_data[CONDITIONS_COLUMN], 
                     label=f'Unit {unit}')

        plt.xlabel('Time')
        plt.ylabel('Operational Condition')
        title = plotTitle("Operational Conditions vs Time", training)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{CONDITION_DIR}/{title.replace(" ", "_")}.png", bbox_inches='tight', dpi=DPI)
        plt.close()
        # From the graph above it is clear that each engine operates at many different operating conditions, not just one.
        # We should try to calculate an average of all previous conditions at each time step for the final result.

        if debug:
            print(df)

        return df, sample_units

    def AddConditionCycleFeatures(df, sample_units, debug=False):
        printBusy("\nAdding cumulative cycles per operational condition", training)
        
        # 2. Cumulative cycles for EACH of the 6 conditions
        for cond in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET]):
            # Create indicator (1 if in this condition, 0 otherwise)
            df[f"TOTAL_COND_{cond}"] = (df[CONDITIONS_COLUMN] == cond).astype(int)
            # Cumulative sum per engine
            df[f"TOTAL_COND_{cond}"] = df.groupby(COLUMN_NAMES[Column.UnitNumber])[f"TOTAL_COND_{cond}"].cumsum()
        
        if debug:
            cols_to_show = [COLUMN_NAMES[Column.UnitNumber], CONDITIONS_COLUMN, 
                           'Cumul_Cycles_Current_Cond'] + \
                          [f'Cumul_Cycles_Cond_{i}' for i in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET])]
            print(df[cols_to_show].head(15))
        
        printBusy("\tPlotting cumulative cycles per condition for sample engines", training)
        
        unit = sample_units[0]
        unit_data = df[df[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
        
        plt.figure(figsize=(12, 6))
        
        # Plot cumulative cycles for each condition
        total_cond_columns = [col for col in df.columns if col.startswith("TOTAL_COND_")]
        for col in total_cond_columns:
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], 
                    unit_data[col], 
                    label=col)
        
        plt.xlabel('Cycle')
        plt.ylabel('Cumulative Cycles')
        title = plotTitle(f'Cumulative Condition Cycles - Unit {unit}', training)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{CONDITION_DIR}/{title.replace(" ", "_")}.png", 
                   bbox_inches='tight', dpi=DPI)
        plt.close()

        # Add condition columns to features:
        feature_columns = [CONDITIONS_COLUMN] + total_cond_columns
        
        return df, feature_columns

    def NormalizePerCondition(df):
        printBusy("\nNormalizing data per condition", training)

        sensor_columns = [COLUMN_NAMES[col] for col in Column if Column.T2.value <= col.value <= Column.W32.value]

        scaler = StandardScaler()
        sensors_normalized = df.groupby(CONDITIONS_COLUMN)[sensor_columns].transform(
            lambda x: scaler.fit_transform(x.to_frame()).flatten())
        sensors_normalized = sensors_normalized.add_suffix("_NORMALIZED")
        df = pd.concat([df, sensors_normalized], axis=1)
        
        return df

    def SmoothSensorData(df, debug=False):
        printBusy("\nSmoothing sensor data", training)

        sensor_columns = [COLUMN_NAMES[col] for col in Column if Column.T2.value <= col.value <= Column.W32.value]

        # From the sensor data plots above we can see that the data is very noisy. We will filter the data using 
        # Exponential Moving Average (EMA)
        # * EMA calculates the average sequentially using current and past cycles. Future values are unknown as in real world prediction.
        # * Places higher weight on the most recent cycles. This helps capture the accelerating degradation curve (exponential wear) typical of turbofan engines as they approach failure.
        for c in sensor_columns:
            norm = f"{c}_NORMALIZED"
            ema_smooth = df.groupby(COLUMN_NAMES[Column.UnitNumber])[norm].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())
            ema_smooth.name = f"{c}_EMA_SMOOTH"
            df = pd.concat([df, ema_smooth], axis=1)

        if debug:
            print(df)
            print(df.describe())

        return df, sensor_columns

    def AddRollingFeatures(df, sensor_columns, feature_columns, debug=False):
        printBusy("\nCreating rolling features", training)
        # Random Forest does not understand time or sequences by itself.
        # It looks at one row at a time and makes a prediction based only on the numbers in that row.
        #
        # Having only the current sensor readings (e.g. temperature, pressure at cycle 150), the model 
        # has no idea whether those values are:
        # * Normal (early in the engine’s life), or
        # * Getting worse (late in life, close to failure).
        # It cannot see the trend or history.
        #
        # Rolling features give the model:
        # * The average value over the last X cycles
        # * Quantifies the recent change in a value
        # * Stability of a value
        #
        # This gives the model context about degradation, which is the key signal for predicting 
        # Remaining Useful Life (RUL).
        #
        # Without these, Random Forest will perform quite poorly. With them, it becomes much 
        # smarter at detecting when an engine is starting to fail.

        for c in sensor_columns:
            smoothed = f"{c}_EMA_SMOOTH"
            roll_mean = df.groupby(COLUMN_NAMES[Column.UnitNumber])[smoothed].transform(lambda x: x.rolling(window=MEAN_WINDOW, min_periods=1).mean())
            roll_std = df.groupby(COLUMN_NAMES[Column.UnitNumber])[smoothed].transform(lambda x: x.rolling(window=STD_WINDOW, min_periods=1).std())

            roll_mean.name = f"{smoothed}_ROLL_MEAN"
            roll_std.name = f"{smoothed}_ROLL_STD"

            df = pd.concat([df, roll_mean], axis=1)
            df = pd.concat([df, roll_std], axis=1)

        # Map sensors relevant to data sets:
        feature_column_map = {
                                #FD001      FD002       FD003       FD004
            Column.BPR:         [True,      True,       True,       True],
            Column.epr:         [False,     True,       True,       True],
            Column.farB:        [False,     True,       False,      True],
            Column.htBleed:     [True,      True,       True,       True],
            Column.Nc:          [True,      True,       True,       True],
            Column.Nf_dmd:      [False,     False,      False,      False],
            Column.Nf:          [True,      True,       True,       True],
            Column.NRc:         [True,      True,       True,       True],
            Column.NRf:         [True,      True,       True,       True],
            Column.P2:          [False,     True,       False,      True],
            Column.P15:         [True,      True,       True,       True],
            Column.P30:         [True,      True,       True,       True],
            Column.PCNfR_dmd:   [False,     True,       False,      False],
            Column.phi:         [True,      True,       True,       True],
            Column.Ps30:        [True,      True,       True,       True],
            Column.T2:          [False,     True,       False,      True],
            Column.T24:         [True,      True,       True,       True],
            Column.T30:         [True,      True,       True,       True],
            Column.T50:         [True,      True,       True,       True],
            Column.W31:         [True,      True,       True,       True],
            Column.W32:         [True,      True,       True,       True],
        }

        for k in feature_column_map.keys():
            if feature_column_map[k][CURRENT_SET.value - 1] == True:
                feature_columns.extend([
                    f"{k.name}_EMA_SMOOTH",
                    f"{k.name}_EMA_SMOOTH_ROLL_MEAN",
                    f"{k.name}_EMA_SMOOTH_ROLL_STD",
                    ])

        return df, feature_columns

    def PlotSensorData(df, sensor_columns, sample_units):
        printBusy("\nPlotting sensor data", training)
        for c in sensor_columns:
            plt.figure(figsize=(20, 20))

            def PopulateSensorSubPlot(plotRows, plotCols, plotIndex, dfSensor, dfSuffix):
                plt.subplot(plotRows, plotCols, plotIndex)
                for unit in sample_units:
                    unit_data = df[df[COLUMN_NAMES[Column.UnitNumber]] == unit]

                    plt.plot(unit_data[RUL_COLUMN if training else COLUMN_NAMES[Column.TimeCycles]], 
                            unit_data[f"{dfSensor}{dfSuffix}"], 
                            label=f"Unit {unit}")
                    plt.title(f"{dfSensor}{dfSuffix} vs RUL")

            # Raw data:
            rows = 3
            cols = 2
            PopulateSensorSubPlot(rows, cols, 1, c, "")
            PopulateSensorSubPlot(rows, cols, 2, c, "_NORMALIZED")
            PopulateSensorSubPlot(rows, cols, 3, c, "_EMA_SMOOTH")
            PopulateSensorSubPlot(rows, cols, 4, c, "_EMA_SMOOTH_ROLL_MEAN")
            PopulateSensorSubPlot(rows, cols, 5, c, "_EMA_SMOOTH_ROLL_STD")
                    
            title = plotTitle(f'Sensor {c} vs. RUL', training)
            plt.suptitle(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{SENSORS_DIR}/{title.replace(" ", "_")}.png", 
                       bbox_inches='tight', dpi=DPI)
            plt.close()


    if training:
        # RUL cannot be calculated for testing sets as we don't know when they fail.
        df = CalculateRUL(df)
        df = ClipRUL(df)

    df, sample_units = ClusterOperationalConditions(df)
    df, feature_columns = AddConditionCycleFeatures(df, sample_units)
    df = NormalizePerCondition(df)
    df, sensor_columns = SmoothSensorData(df)
    df, feature_columns = AddRollingFeatures(df, sensor_columns, feature_columns)
    PlotSensorData(df, sensor_columns, sample_units)

    return df, feature_columns

def TrainTestSplit(df_train, debug=False):
    printHeading("Train/Test Split", training=False)
    unique_units = df_train[COLUMN_NAMES[Column.UnitNumber]].unique()
    train_units, test_units = train_test_split(unique_units, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    df_train_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(train_units)].copy()
    df_test_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(test_units)].copy()
    print(f"Train engines: {len(train_units)}, Test engines: {len(test_units)}")
    return df_train_split, df_test_split

# NASA's score (common for RUL)
def rul_score(y_test, y_pred):
    diff = y_pred - y_test
    score = np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))
    return abs(score)
CMAPSS_SCORER = make_scorer(rul_score, greater_is_better=False)

def HyperparameterTuning(df_train, feature_columns, debug=False):
    printHeading("Hyperparameter Tuning", training=True)

    ## Extract X and Y data:
    print(f"Target column: {RUL_CLIPPED_COLUMN}")
    print(f"Feature columns: {feature_columns}")

    X = df_train[feature_columns].values
    y = df_train[RUL_CLIPPED_COLUMN].values

    search = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=PARAM_GRIDS[CURRENT_SET],
            cv=2,
            verbose=3,
            scoring=CMAPSS_SCORER,
            n_jobs=-1)
    search.fit(X, y)

    test = search.score(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Estimator:", search.best_estimator_)
    print("Test Score:", test)

    return

def RandomForestModel(df_train_split, feature_columns, debug=False):
    printHeading("Random Forest Model", training=True)

    ## Extract X and Y data:
    target_col = RUL_CLIPPED_COLUMN
    feature_cols = feature_columns
    
    print(f"Target column: {RUL_CLIPPED_COLUMN}")
    print(f"Feature columns: {feature_cols}")

    X_train = df_train_split[feature_cols].values
    y_train = df_train_split[target_col].values

    print(f"Training the random forest model...")
    model = RandomForestRegressor(
            n_estimators=NUM_TREES,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            min_samples_split=MIN_SAMPLES_SPLIT,
            max_depth=MAX_DEPTH,
            n_jobs=-1,
            random_state=RANDOM_STATE
            )

    model.fit(X_train, y_train)

    feature_importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("Feature importances:")
    print(feature_importances.sort_values(ascending=False).head(20))

    return model, feature_cols

def EvaluateModel(df_train_split, df_test_split, df_test, model, feature_cols, debug=False):
    print("Evaluating Random Forest Model")

    def PopulateScatterSubPlot(plotRows, plotCols, plotIndex, y, y_pred, description, rmse, nasa):
        plt.subplot(plotRows, plotCols, plotIndex)
        plt.scatter(y, y_pred, alpha=0.5, s=10)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Perfect prediction line
        plt.text(0, 125, f"RMSE={rmse:.2f}\nNASA={int(nasa)}", color='r', fontweight='bold')
        plt.xlabel(f"Actual RUL (Clipped @ {RUL_LIMIT})")
        plt.ylabel('Predicted RUL')
        plt.title(f'Actual vs Predicted RUL ({description})')
        plt.grid(True)

    def PopulatePredictedRULvsTime(plotRows, plotCols, plotIndex, df, sample_units, actual_avail: bool, description):
        plt.subplot(plotRows, plotCols, plotIndex)
        for unit in sample_units:
            unit_data = df[df[COLUMN_NAMES[Column.UnitNumber]] == unit]

            if actual_avail:
                actual = unit_data[RUL_CLIPPED_COLUMN]
                plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], actual, label=f'Unit {unit} - Actual', linestyle='-')#, marker='o')

            pred = model.predict(unit_data[feature_cols].values) - NASA_SAFETY_BUFFER
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], pred, label=f'Unit {unit} - Predicted', linestyle='--')
            plt.xlabel('Time (Cycles)')
            plt.ylabel('RUL')
            plt.title(f'RUL Prediction over Time ({description})')
            plt.legend()
            plt.grid(True)

    # Scatter plot
    plt.figure(figsize=(20, 20))
    rows = 3
    cols = 2
    i = 1

    for case in [(df_train_split, f"TRAIN SPLIT {CURRENT_SET}", True), (df_test_split, f"TEST SPLIT {CURRENT_SET}", True), (df_test, f"NASA TEST DATA {CURRENT_SET}", False)]:
        df = case[0]
        description = case[1]
        rul_avail = case[2]

        X = df[feature_cols].values
        y_pred = model.predict(X) - NASA_SAFETY_BUFFER
        if rul_avail:
            y = df[RUL_CLIPPED_COLUMN].values

        else:
            # Read testing RUL data and remove blank column:
            df_test_rul = pd.read_csv(testing_RUL_files[CURRENT_SET], sep=' ', header=None).iloc[:, :1]
            df_test["RUL_PRED"] = y_pred
            df_test_max = df_test.loc[df_test.groupby(COLUMN_NAMES[Column.UnitNumber])[COLUMN_NAMES[Column.TimeCycles]].idxmax()]
            y_pred = df_test_max["RUL_PRED"].reset_index(drop=True)
            y = df_test_rul.iloc[:, 0].clip(upper=RUL_LIMIT)

            print(y_pred)
            print(y)
            

        ## Metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"RMSE ({description}): {rmse:.2f}")
        nasa = rul_score(y, y_pred)
        print(f"NASA Score ({description}): {nasa:.2f}")
        PopulateScatterSubPlot(rows, cols, i, y, y_pred, description, rmse, nasa)
        i = i+1

        sample_units = random.sample(list(df[COLUMN_NAMES[Column.UnitNumber]].unique()), 5)
        PopulatePredictedRULvsTime(rows, cols, i, df, sample_units, rul_avail, description)
        i = i + 1

    plt.savefig(f"{RESULTS_DIR}/Evaluation_Results_{CURRENT_SET.name}.png", 
               bbox_inches='tight', dpi=DPI)
    plt.close()


df_train = LoadData(training=True)
df_test = LoadData(training=False)

df_train = PrepareData(df_train, training=True)
df_test = PrepareData(df_test, training=False)

df_train, feature_columns = FeatureEngineering(df_train, training=True)
df_test, _ = FeatureEngineering(df_test, training=False)

df_train_split, df_test_split = TrainTestSplit(df_train)

if TUNE:
    HyperparameterTuning(df_train, feature_columns)
else:
    model, feature_cols = RandomForestModel(df_train_split, feature_columns)
    EvaluateModel(df_train_split, df_test_split, df_test, model, feature_cols)
