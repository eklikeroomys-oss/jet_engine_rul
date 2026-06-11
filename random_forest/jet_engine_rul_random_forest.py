# Jet Engine RUL Using NASA CMAPSS Data
# "Data URL: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

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

CURRENT_SET = Set.FD001
#TUNE = True
TUNE = False

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

NASA_SAFETY_BUFFER = 0 # Force model to predict a bit earlier
RUL_LIMIT = 130

RANDOM_STATE = 42
PARAM_GRID = {
    'n_estimators': [450],
    'max_depth': [10],
    'min_samples_leaf': [4],
    'min_samples_split': range(2, 20, 2),
    'max_features': ['sqrt'],
    'random_state': [RANDOM_STATE],
    'n_jobs': [-1]
}


training_files = {
        Set.FD001: "../Data/train_FD001.txt",
        Set.FD002: "../Data/train_FD002.txt",
        Set.FD003: "../Data/train_FD003.txt",
        Set.FD004: "../Data/train_FD004.txt",
        }

OUTPUT_DIR = Path(f"output/{CURRENT_SET.name}")

RUL_DIR = Path(f"{OUTPUT_DIR}/1_RUL")
CONDITION_DIR = Path(f"{OUTPUT_DIR}/2_Conditions")
SENSORS_DIR = Path(f"{OUTPUT_DIR}/3_Sensors")
RESULTS_DIR = Path(f"{OUTPUT_DIR}/Results")

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

# Feature Engineered Columns:
RATIO_T24_T2 = "T24_vs_T2"
RATIO_T30_T24 = "T30_vs_T24"
RATIO_T50_T30 = "T50_vs_T30"
RATIO_T50_T2 = "T50_vs_T2"
RATIO_W32_W31 = "W32_vs_W31"

RUL_COLUMN = "RUL"
RUL_CLIPPED_COLUMN = "RUL_CLIPPED"
CONDITIONS_COLUMN = "Operational Condition"
TOTAL_COND_1 = "Total Cycles Condition 1"
TOTAL_COND_2 = "Total Cycles Condition 2"
TOTAL_COND_3 = "Total Cycles Condition 3"
TOTAL_COND_4 = "Total Cycles Condition 4"
TOTAL_COND_5 = "Total Cycles Condition 5"
TOTAL_COND_6 = "Total Cycles Condition 6"

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

def printHeading(heading):
    c = "#"
    length = len(heading)
    print(f"\n{c * (length + 4)}")
    print(f"{c} {heading} {c}")
    print(f"{c * (length + 4)}")

def LoadData(debug=False):
    printHeading("Data Gathering")
    head_count = 5
    print(f"Loading training file {CURRENT_SET.name}...")
    df_train = pd.read_csv(training_files[CURRENT_SET], sep=' ', header=None)
    if debug: 
        print(df_train.head(head_count))
        print(df_train.describe())

    return df_train

def PlotHistogram(data, bins, dir, title, xlabel, ylabel):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{dir}/{title.replace(" ", "_")}.png", bbox_inches='tight', dpi=DPI)
    plt.close()

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

    print("Sorting data by unit number and time (cycles)...")
    # Ensure data is sorted
    df_train = df_train.sort_values([COLUMN_NAMES[Column.UnitNumber], 
                                   COLUMN_NAMES[Column.TimeCycles]]).reset_index(drop=True)
    if debug:
        print(df_train)
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

    def ClusterOperationalConditions(df_train, debug=False):
        print("\nClustering the operational conditions...")
        df_operational_params = df_train[OPERATIONAL_PARAMS].copy()

        print("\tScaling condition data before KMeans fit")
        operational_params_scaler = StandardScaler()
        df_operational_params_scaled = operational_params_scaler.fit_transform(df_operational_params)

        print("\tPerforming KMeans fit")
        km = KMeans(n_clusters=NUM_OPERATIONAL_CONDITIONS[CURRENT_SET])
        df_train[CONDITIONS_COLUMN] = km.fit_predict(df_operational_params_scaled)
        print("\t\tCluster centers :")
        operational_condition_centers = operational_params_scaler.inverse_transform(km.cluster_centers_)
        for i in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET]):
            print(f"\t\tCondition {i+1}: {OPERATIONAL_PARAMS[0]} = {int(operational_condition_centers[i][0] * 1000)}Ft, \
                    {OPERATIONAL_PARAMS[1]} = {round(operational_condition_centers[i][1], 2)}, {OPERATIONAL_PARAMS[2]} = {round(operational_condition_centers[i][2], 2)}")

        if NUM_OPERATIONAL_CONDITIONS[CURRENT_SET] > 1:
            print("\tCalculating operating condition clusters silhouette score...")
            print(f"\t\tResult: {silhouette_score(df_operational_params_scaled, df_train[CONDITIONS_COLUMN], metric="euclidean", sample_size=SILHOUETTE_SCORE_SAMPLE_SIZE, random_state=RANDOM_STATE)}")

        print("\tPlotting clusters...")
        operational_conditions_fig = plt.figure(figsize=(10,8))
        ax = operational_conditions_fig.add_subplot(111, projection='3d')
        operational_conditions_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
        for cluster in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET]):
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
        plt.savefig(f"{CONDITION_DIR}/Operational_Condition_Clusters.png", bbox_inches='tight', dpi=DPI)
        plt.close()

        print("\tPlotting operational condition over cycles for a few sample engines...")
        sample_units = random.sample(list(df_train[COLUMN_NAMES[Column.UnitNumber]].unique()), SAMPLE_UNITS)
        plt.figure(figsize=(12, 8))
        for unit in sample_units:
            unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], 
                     unit_data[CONDITIONS_COLUMN], 
                     label=f'Unit {unit}')

        plt.xlabel('Cycle')
        plt.ylabel('Operational Condition')
        plt.title('Operational Condition vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{CONDITION_DIR}/Operational_Condition_vs_Time.png", bbox_inches='tight', dpi=DPI)
        plt.close()
        # From the graph above it is clear that each engine operates at many different operating conditions, not just one.
        # We should try to calculate an average of all previous conditions at each time step for the final result.

        if debug:
            print(df_train)

        return df_train, sample_units

    def AddConditionCycleFeatures(df_train, sample_units, debug=False):
        print("\nAdding cumulative cycles per operational condition...")
        
        # 2. Cumulative cycles for EACH of the 6 conditions
        for cond in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET]):
            # Create indicator (1 if in this condition, 0 otherwise)
            df_train[f"TOTAL_COND_{cond}"] = (df_train[CONDITIONS_COLUMN] == cond).astype(int)
            # Cumulative sum per engine
            df_train[f"TOTAL_COND_{cond}"] = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[f"TOTAL_COND_{cond}"].cumsum()
        
        if debug:
            cols_to_show = [COLUMN_NAMES[Column.UnitNumber], CONDITIONS_COLUMN, 
                           'Cumul_Cycles_Current_Cond'] + \
                          [f'Cumul_Cycles_Cond_{i}' for i in range(NUM_OPERATIONAL_CONDITIONS[CURRENT_SET])]
            print(df_train[cols_to_show].head(15))
        
        print("\tPlotting cumulative cycles per condition for sample engines...")
        
        unit = sample_units[0]
        unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
        
        plt.figure(figsize=(12, 6))
        
        # Plot cumulative cycles for each condition
        total_cond_columns = [col for col in df_train.columns if col.startswith("TOTAL_COND_")]
        for col in total_cond_columns:
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], 
                    unit_data[col], 
                    label=col)
        
        plt.xlabel('Cycle')
        plt.ylabel('Cumulative Cycles')
        plt.title(f'Cumulative Condition Cycles - Unit {unit}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{CONDITION_DIR}/Cumulative_Condition_Cycles.png", 
                   bbox_inches='tight', dpi=DPI)
        plt.close()

        # Add condition columns to features:
        feature_columns = [CONDITIONS_COLUMN] + total_cond_columns
        
        return df_train, feature_columns

    def NormalizePerCondition(df_train):
        print("\nNormalizing data per condition...")

        sensor_columns = [COLUMN_NAMES[col] for col in Column if Column.T2.value <= col.value <= Column.W32.value]

        scaler = StandardScaler()
        sensors_normalized = df_train.groupby(CONDITIONS_COLUMN)[sensor_columns].transform(
            lambda x: scaler.fit_transform(x.to_frame()).flatten())
        sensors_normalized = sensors_normalized.add_suffix("_NORMALIZED")
        df_train = pd.concat([df_train, sensors_normalized], axis=1)
        
        return df_train

    def SmoothSensorData(df_train, sample_units, debug=False):
        print("\nSmoothing and plotting sensor data...")

        sensor_columns = [COLUMN_NAMES[col] for col in Column if Column.T2.value <= col.value <= Column.W32.value]

        # From the sensor data plots above we can see that the data is very noisy. We will filter the data using 
        # Exponential Moving Average (EMA)
        # * EMA calculates the average sequentially using current and past cycles. Future values are unknown as in real workd prediction.
        # * Places higher weight on the most recent cycles. This helps capture the accelerating degradation curve (exponential wear) typical of turbofan engines as they approach failure.
        print("\tSmoothing sensor data to remove noise...")
        for c in sensor_columns:
            norm = f"{c}_NORMALIZED"
            ema_smooth = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[norm].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())
            ema_smooth.name = f"{c}_EMA_SMOOTH"
            df_train = pd.concat([df_train, ema_smooth], axis=1)

        if debug:
            print(df_train)
            print(df_train.describe())

        return df_train, sensor_columns

    def AddRollingFeatures(df_train, sensor_columns, feature_columns, sample_units, debug=False):
        print("\nCreating rolling features...")
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

        print("\tCalculating rolling features...")
        for c in sensor_columns:
            smoothed = f"{c}_EMA_SMOOTH"
            roll_mean = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[smoothed].transform(lambda x: x.rolling(window=MEAN_WINDOW, min_periods=1).mean())
            roll_std = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[smoothed].transform(lambda x: x.rolling(window=STD_WINDOW, min_periods=1).std())

            roll_mean.name = f"{smoothed}_ROLL_MEAN"
            roll_std.name = f"{smoothed}_ROLL_STD"

            df_train = pd.concat([df_train, roll_mean], axis=1)
            df_train = pd.concat([df_train, roll_std], axis=1)

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

        return df_train, feature_columns

    def PlotSensorData(df_train, sensor_columns, sample_units):
        print("\nPlotting sensor data...")
        for c in sensor_columns:
            plt.figure(figsize=(20, 20))

            def PopulateSensorSubPlot(plotRows, plotCols, plotIndex, dfSensor, dfSuffix):
                plt.subplot(plotRows, plotCols, plotIndex)
                for unit in sample_units:
                    unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit]

                    plt.plot(unit_data[RUL_COLUMN], 
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
                    
            plt.suptitle(f'Sensor {c} vs. RUL')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{SENSORS_DIR}/Sensor_{c}_vs_RUL.png", 
                       bbox_inches='tight', dpi=DPI)
            plt.close()


    df_train = CalculateRUL(df_train)
    df_train = ClipRUL(df_train)
    df_train, sample_units = ClusterOperationalConditions(df_train)
    df_train, feature_columns = AddConditionCycleFeatures(df_train, sample_units)
    df_train = NormalizePerCondition(df_train)
    df_train, sensor_columns = SmoothSensorData(df_train, sample_units)
    df_train, feature_columns = AddRollingFeatures(df_train, sensor_columns, feature_columns, sample_units)
    PlotSensorData(df_train, sensor_columns, sample_units)

    return df_train, feature_columns, sample_units

def TrainTestSplit(df_train, debug=False):
    printHeading("Train/Test Split")
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
    printHeading("Hyperparameter Tuning")

    ## Extract X and Y data:
    print(f"Target column: {RUL_CLIPPED_COLUMN}")
    print(f"Feature columns: {feature_columns}")

    X = df_train[feature_columns].values
    y = df_train[RUL_CLIPPED_COLUMN].values

    search = GridSearchCV(
            estimator=RandomForestRegressor(), 
            param_grid=PARAM_GRID, 
            cv=2, 
            verbose=3,
            scoring=CMAPSS_SCORER)
    search.fit(X, y)

    test = search.score(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Estimator:", search.best_estimator_)
    print("Test Score:", test)

    return

def RandomForestModel(df_train_split, df_test_split, feature_columns, debug=False):
    printHeading("Random Forest Model")

    ## Extract X and Y data:
    target_col = RUL_CLIPPED_COLUMN
    feature_cols = feature_columns
    
    print(f"Target column: {RUL_CLIPPED_COLUMN}")
    print(f"Feature columns: {feature_cols}")

    X_train = df_train_split[feature_cols].values
    y_train = df_train_split[target_col].values

    X_test = df_test_split[feature_cols].values
    y_test = df_test_split[target_col].values

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
    print("Feature imporances:")
    print(feature_importances.sort_values(ascending=False).head(20))

    return model, feature_cols, X_test, y_test, X_train, y_train

def EvaluateModel(df, model, X, y, feature_cols, identifier, debug=True):
    printHeading("Evaluating Random Forest Model")
    y_pred = model.predict(X) - NASA_SAFETY_BUFFER

    ## Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE ({identifier}): {rmse:.2f}")

    print(f"NASA Score ({identifier}): {rul_score(y, y_pred):.2f}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Perfect prediction line

    plt.xlabel('Actual RUL (Clipped)')
    plt.ylabel('Predicted RUL')
    plt.title(f'Actual vs Predicted RUL ({identifier})')
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/Predicted_vs_Actual_RUL_{identifier.replace(" ", "_")}.png", 
               bbox_inches='tight', dpi=DPI)
    plt.close()

    plt.figure(figsize=(12, 8))
    sample_units = random.sample(list(df[COLUMN_NAMES[Column.UnitNumber]].unique()), 5)
    for unit in sample_units:
        unit_data = df[df[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()

        actual = unit_data[RUL_CLIPPED_COLUMN]
        pred = model.predict(unit_data[feature_cols].values) - NASA_SAFETY_BUFFER

        plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], actual, label=f'Unit {unit} - Actual', linestyle='-', marker='o')
        plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], pred, label=f'Unit {unit} - Predicted', linestyle='--')

    plt.xlabel('Cycle')
    plt.ylabel('RUL')
    plt.title('RUL Prediction over Cycles for Sample Engines')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/Sample_Engine_Predictions_{identifier.replace(" ", "_")}.png", 
               bbox_inches='tight', dpi=DPI)
    plt.close()

df_train = LoadData()
df_train = PrepareData(df_train)
df_train, feature_columns, sample_units = FeatureEngineering(df_train)
df_train_split, df_test_split = TrainTestSplit(df_train)

if TUNE:
    HyperparameterTuning(df_train, feature_columns)
else:
    model, feature_cols, X_test, y_test, X_train, y_train = RandomForestModel(df_train_split, df_test_split, feature_columns)
    EvaluateModel(df_train_split, model, X_train, y_train, feature_cols, "Training Split")
    EvaluateModel(df_test_split, model, X_test, y_test, feature_cols, "Testing Split")
