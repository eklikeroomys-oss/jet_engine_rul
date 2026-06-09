# Jet Engine RUL Using NASA CMAPSS Data
# "Data URL: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns

from enum import Enum
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from pandas.core.window import rolling
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
TOTAL_COND_CYCLE_COLS = [
        TOTAL_COND_1,
        TOTAL_COND_2,
        TOTAL_COND_3,
        TOTAL_COND_4,
        TOTAL_COND_5,
        TOTAL_COND_6
        ]

# Constants from given data
NUM_OPERATIONAL_CONDITIONS = 6
OPERATIONAL_PARAMS = [COLUMN_NAMES[Column.Altitude], COLUMN_NAMES[Column.MachNumber], COLUMN_NAMES[Column.TRA]]

# Parameter Tuning:
WINDOW_SIZE = 30
TRAINING_FILE_COUNT = 4
RANDOM_STATE = 42
SILHOUETTE_SCORE_SAMPLE_SIZE = 5000
SENSOR_VARIANCE_LIMIT = 100
RUL_LIMIT = 150
SAMPLE_UNITS = 25

TEST_SIZE = 0.3

EMA_SPAN = 15 # EMA span for filtering out white noise without flattening critical curves near EOL.
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 2
NUM_TREES = 50
MAX_DEPTH = None

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
        # Engines start failing at about 128 cycles, so we should be able to clip RUL <= RUL_LIMIT cycles.
        return df_train

    def ClipRUL(df_train, debug=False):
        rul_limit = RUL_LIMIT
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

    def ClusterOperationalConditions(df_train, debug=False):
        print("\nClustering the operational conditions...")
        df_operational_params = df_train[OPERATIONAL_PARAMS].copy()

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
        print(f"\t\tResult: {silhouette_score(df_operational_params_scaled, df_train[CONDITIONS_COLUMN], metric="euclidean", sample_size=SILHOUETTE_SCORE_SAMPLE_SIZE, random_state=RANDOM_STATE)}")

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
        plt.title('Operational Condition over Time for Sample Engines')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/Operational_Condition_Per_Engine_Sample.png", bbox_inches='tight', dpi=300)
        plt.close()
        # From the graph above it is clear that each engine operates at many different operating conditions, not just one.
        # We should try to calculate an average of all previous conditions at each time step for the final result.

        if debug:
            print(df_train)

        return df_train, sample_units

    def AddConditionCycleFeatures(df_train, sample_units, debug=False):
        print("\nAdding cumulative cycles per operational condition...")
        
        # 2. Cumulative cycles for EACH of the 6 conditions
        for cond in range(NUM_OPERATIONAL_CONDITIONS):
            # Create indicator (1 if in this condition, 0 otherwise)
            df_train[TOTAL_COND_CYCLE_COLS[cond]] = (df_train[CONDITIONS_COLUMN] == cond).astype(int)
            # Cumulative sum per engine
            df_train[TOTAL_COND_CYCLE_COLS[cond]] = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[TOTAL_COND_CYCLE_COLS[cond]].cumsum()
        
        if debug:
            cols_to_show = [COLUMN_NAMES[Column.UnitNumber], CONDITIONS_COLUMN, 
                           'Cumul_Cycles_Current_Cond'] + \
                          [f'Cumul_Cycles_Cond_{i}' for i in range(NUM_OPERATIONAL_CONDITIONS)]
            print(df_train[cols_to_show].head(15))
        
        print("\tPlotting cumulative cycles per condition for sample engines...")
        
        unit = sample_units[0]
        unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
        
        plt.figure(figsize=(12, 6))
        
        # Plot cumulative cycles for each condition
        for col in TOTAL_COND_CYCLE_COLS:
            plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], 
                    unit_data[col], 
                    label=col)
        
        # Also plot total cycles for reference
        # plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], 
        #         unit_data[COLUMN_NAMES[Column.TimeCycles]], 
        #         label='Total Cycles', linestyle='--', color='black', alpha=0.7)
        
        plt.xlabel('Cycle')
        plt.ylabel('Cumulative Cycles')
        plt.title(f'Cumulative Cycles per Condition - Unit {unit}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/Cumulative_Condition_Cycles_Sample_Unit.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
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
            ema_smooth = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].transform(lambda x: x.ewm(span=EMA_SPAN, adjust=False).mean())
            ema_smooth.name = f"{c}_EMA_SMOOTH"
            df_train = pd.concat([df_train, ema_smooth], axis=1)

            plt.figure(figsize=(20, 20))
            plt.subplot(2, 1, 1)
            for unit in sample_units:
                unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
                plt.plot(unit_data[RUL_COLUMN], 
                        unit_data[f"{c}"], 
                        label=f"Unit {unit}")
                plt.title(f"{c} Raw Data")

            plt.subplot(2, 1, 2)
            for unit in sample_units:
                unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
                plt.plot(unit_data[RUL_COLUMN], 
                        unit_data[f"{c}_EMA_SMOOTH"], 
                        label=f"Unit {unit}")
                plt.title(f"{c} Smoothed Data")
                    
            plt.suptitle(f'Sensor {c} vs. RUL')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{OUTPUT_DIR}/Sensor_{c}_vs_RUL.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()


        # The following smoothed sensor data looks promising, add them as features:
        feature_columns = [
                "farB_EMA_SMOOTH",
                "htBleed_EMA_SMOOTH",
                "Nc_EMA_SMOOTH",
                "NRc_EMA_SMOOTH",
                "Ps30_EMA_SMOOTH",
                "T24_EMA_SMOOTH",
                "T30_EMA_SMOOTH",
                "T50_EMA_SMOOTH",
                ]

        if debug:
            print(df_train)
            print(df_train.describe())

        return df_train, sensor_columns, feature_columns

    def AddRatioFeatures(df_train, sensor_columns, feature_columns, sample_units, debug=False):
        print("\nAdding some ratio sensors...")
        df_train[RATIO_T24_T2] = df_train[f"{COLUMN_NAMES[Column.T24]}_EMA_SMOOTH"] / df_train[f"{COLUMN_NAMES[Column.T2]}_EMA_SMOOTH"]
        df_train[RATIO_T30_T24] = df_train[f"{COLUMN_NAMES[Column.T30]}_EMA_SMOOTH"] / df_train[f"{COLUMN_NAMES[Column.T24]}_EMA_SMOOTH"]
        df_train[RATIO_T50_T30] = df_train[f"{COLUMN_NAMES[Column.T50]}_EMA_SMOOTH"] / df_train[f"{COLUMN_NAMES[Column.T30]}_EMA_SMOOTH"]
        df_train[RATIO_T50_T2] = df_train[f"{COLUMN_NAMES[Column.T50]}_EMA_SMOOTH"] / df_train[f"{COLUMN_NAMES[Column.T2]}_EMA_SMOOTH"]
        df_train[RATIO_W32_W31] = df_train[f"{COLUMN_NAMES[Column.W32]}_EMA_SMOOTH"] / df_train[f"{COLUMN_NAMES[Column.W31]}_EMA_SMOOTH"]

        ratio_columns = [RATIO_W32_W31, RATIO_T50_T2, RATIO_T50_T30, RATIO_T30_T24, RATIO_T24_T2]

        # Plot sensor data for a few units:
        print("\tPlotting ratio data...")
        for s in ratio_columns:
            plt.figure(figsize=(12, 6))
            for unit in sample_units:
                unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
                plt.plot(unit_data[RUL_COLUMN], 
                         unit_data[s],
                         label=f"Unit {unit}")
                
            plt.xlabel('RUL')
            plt.ylabel(f'{s}')
            plt.title(f'{s} vs. RUL')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{OUTPUT_DIR}/Ratio_{s}_vs_RUL.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()

        # Some of the ratio features look promising:
        sensor_columns.extend([RATIO_T50_T30, RATIO_T50_T2, RATIO_T30_T24, RATIO_T24_T2])
        feature_columns.extend([RATIO_T50_T30, RATIO_T50_T2, RATIO_T30_T24, RATIO_T24_T2])

        if debug:
            print(df_train)
            print(df_train.describe())

        return df_train, sensor_columns, feature_columns

    def CreateRollingFeatures(df_train, sensor_columns, feature_columns, sample_units, debug=False):
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

        # Calculate slope:
        def analytical_slope(y):
            n = len(y)
            x = np.arange(n)

            x_mean = (n - 1) / 2.0
            y_mean = np.mean(y)

            # Sum of products minus correction factor
            covariance = np.sum(x * y) - n * x_mean * y_mean
            variance = (n * (n**2 - 1)) / 12.0  # Analytical variance of arange(n)

            if variance == 0:
                return 0.0

            return covariance / variance

        print("\tCalculating rolling features for operational parameters...")
        for c in OPERATIONAL_PARAMS:
            roll_mean = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean())
            roll_mean.name = f"{c}_ROLL_MEAN"
            df_train = pd.concat([df_train, roll_mean], axis=1)


        print("\tCalculating rolling features for sensors...")
        for c in sensor_columns:
            cumu_min = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].cummin()
            cumu_max = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].cummax()
            roll_min = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).min())
            roll_max = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).max())
            roll_mean = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean())
            roll_std = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).std())
            roll_var = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[c].transform(lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).var())

            cumu_min.name = f"{c}_CUMU_MIN"
            cumu_max.name = f"{c}_CUMU_MAX"
            roll_min.name = f"{c}_ROLL_MIN"
            roll_max.name = f"{c}_ROLL_MAX"
            roll_mean.name = f"{c}_ROLL_MEAN"
            roll_std.name = f"{c}_ROLL_STD"
            roll_var.name = f"{c}_ROLL_VAR"

            df_train = pd.concat([df_train, cumu_min], axis=1)
            df_train = pd.concat([df_train, cumu_max], axis=1)
            df_train = pd.concat([df_train, roll_min], axis=1)
            df_train = pd.concat([df_train, roll_max], axis=1)
            df_train = pd.concat([df_train, roll_mean], axis=1)
            df_train = pd.concat([df_train, roll_std], axis=1)
            df_train = pd.concat([df_train, roll_var], axis=1)


        print("\tCalculating rolling slopes for some rolling features...")
        for c in sensor_columns:
            roll_mean_slope = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[f"{c}_ROLL_MEAN"].transform(lambda x: x.rolling(window=10, min_periods=1).apply(analytical_slope, raw=True))
            roll_max_slope = df_train.groupby(COLUMN_NAMES[Column.UnitNumber])[f"{c}_ROLL_MAX"].transform(lambda x: x.rolling(window=10, min_periods=1).apply(analytical_slope, raw=True))
            roll_mean_slope.name = f"{c}_ROLL_MEAN_SLOPE"
            roll_max_slope.name = f"{c}_ROLL_MAX_SLOPE"
            df_train = pd.concat([df_train, roll_mean_slope], axis=1)
            df_train = pd.concat([df_train, roll_max_slope], axis=1)

        # Plot rolling sensor data for a few units:
        print("\tPlotting rolling sensor features...")
        for i in range(len(sensor_columns)):
            s = sensor_columns[i]
            plt.figure(figsize=(20, 20))
            rolling_postfixes = ["", "_CUMU_MIN", "_CUMU_MAX", "_ROLL_MIN", "_ROLL_MAX", "_ROLL_MEAN", "_ROLL_STD", "_ROLL_VAR", "_ROLL_MEAN_SLOPE", "_ROLL_MAX_SLOPE"]
            for j in range(len(rolling_postfixes)):
                plt.subplot(6, 2, j+1)
                for unit in sample_units:
                    unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
                    #plt.plot(unit_data[unit_data[RUL_CLIPPED_COLUMN] < RUL_LIMIT][RUL_CLIPPED_COLUMN], 
                            #unit_data[unit_data[RUL_CLIPPED_COLUMN] < RUL_LIMIT][f"{s}{rolling_postfixes[j]}"], 
                    plt.plot(unit_data[RUL_COLUMN], 
                            unit_data[f"{s}{rolling_postfixes[j]}"], 
                            label=f"Unit {unit}")
                    plt.title(f"{s}{rolling_postfixes[j]}")
                    
            plt.suptitle(f'Rolling Sensor {s} vs. RUL')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{OUTPUT_DIR}/Rolling_Sensor_{s}_vs_Time.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()

        print("\tPlotting rolling parameter features...")
        for i in range(len(OPERATIONAL_PARAMS)):
            s = OPERATIONAL_PARAMS[i]
            plt.figure(figsize=(20, 12))
            rolling_postfixes = ["", "_ROLL_MEAN"]
            for j in range(len(rolling_postfixes)):
                plt.subplot(2, 1, j+1)
                for unit in sample_units:
                    unit_data = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()
                    #plt.plot(unit_data[unit_data[RUL_CLIPPED_COLUMN] < RUL_LIMIT][RUL_CLIPPED_COLUMN], 
                            #unit_data[unit_data[RUL_CLIPPED_COLUMN] < RUL_LIMIT][f"{s}{rolling_postfixes[j]}"], 
                    plt.plot(unit_data[RUL_COLUMN], 
                            unit_data[f"{s}{rolling_postfixes[j]}"], 
                            label=f"Unit {unit}")
                    plt.title(f"{s}{rolling_postfixes[j]}")
                    
            plt.suptitle(f'Rolling Param {s} vs. RUL')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{OUTPUT_DIR}/Rolling_Parameter_{s}_vs_Time.png", 
                       bbox_inches='tight', dpi=300)
            plt.close()

        # The following rolling features look promising:
        feature_columns = [
                "farB_ROLL_STD",
                "farB_ROLL_MEAN",
                "farB_ROLL_VAR",

                "NRc_CUMU_MAX",
                "NRc_ROLL_MAX",
                "NRc_ROLL_MEAN",
                "NRc_ROLL_MAX_SLOPE",

                "NRf_CUMU_MAX",
                "NRf_ROLL_MAX",

                "Ps30_CUMU_MAX",
                #"Ps30_ROLL_MAX",  <-- Overdominant
                "Ps30_ROLL_MEAN",
                "Ps30_ROLL_MEAN_SLOPE",

                "T50_vs_T2_CUMU_MAX",
                "T50_vs_T2_ROLL_MAX",
                "T50_vs_T2_ROLL_MEAN",

                "T50_vs_T30_CUMU_MAX",
                "T50_vs_T30_ROLL_MAX",
                "T50_vs_T30_ROLL_MEAN",

                "T50_CUMU_MAX",
                "T50_ROLL_MAX",
                "T50_ROLL_MEAN",
                ]
                

        return df_train, feature_columns


    df_train = CalculateRUL(df_train)
    df_train = ClipRUL(df_train)
    df_train, sample_units = ClusterOperationalConditions(df_train)
    df_train = AddConditionCycleFeatures(df_train, sample_units)
    df_train, sensor_columns, feature_columns = SmoothSensorData(df_train, sample_units)
    df_train, sensor_columns, feature_columns = AddRatioFeatures(df_train, sensor_columns, feature_columns, sample_units)
    #df_train, feature_columns = CreateRollingFeatures(df_train, sensor_columns, feature_columns, sample_units)

    print(feature_columns)
    return df_train, feature_columns, sample_units

def TrainTestSplit(df_train, debug=False):
    printHeading("Train/Test Split")
    unique_units = df_train[COLUMN_NAMES[Column.UnitNumber]].unique()
    train_units, test_units = train_test_split(unique_units, test_size=0.40, random_state=RANDOM_STATE)
    df_train_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(train_units)].copy()
    df_test_split = df_train[df_train[COLUMN_NAMES[Column.UnitNumber]].isin(test_units)].copy()
    print(f"Train engines: {len(train_units)}, Test engines: {len(test_units)}")
    return df_train_split, df_test_split

def RandomForestModel(df_train_split, df_test_split, feature_columns, debug=False):
    printHeading("Random Forest Model")

    ## Extract X and Y data:
    target_col = RUL_CLIPPED_COLUMN
    feature_cols = feature_columns + [CONDITIONS_COLUMN] + TOTAL_COND_CYCLE_COLS
    
    print(f"Target column: {RUL_CLIPPED_COLUMN}")
    print(f"Feature columns: {feature_cols}")

    X_train = df_train_split[feature_cols]
    y_train = df_train_split[target_col]

    X_test = df_test_split[feature_cols]
    y_test = df_test_split[target_col]

    print(f"Training the random forest model...")
    model = RandomForestRegressor(
            n_estimators=NUM_TREES,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            max_depth=MAX_DEPTH,
            n_jobs=-1,
            )

    model.fit(X_train, y_train)

    feature_importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("Feature imporances:")
    print(feature_importances.sort_values(ascending=False).head(20))

    return model, feature_cols, X_test, y_test, X_train, y_train

def EvaluateModel(df, model, X, y, feature_cols, identifier, debug=True):
    printHeading("Evaluating Random Forest Model")
    y_pred = model.predict(X)

    ## Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE ({identifier}): {rmse:.2f}")

    # NASA's score (common for RUL)
    def rul_score(y_test, y_pred):
        diff = y_pred - y_test
        score = np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))
        return score
    print(f"NASA Score ({identifier}): {rul_score(y, y_pred):.2f}")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, s=10)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Perfect prediction line

    plt.xlabel('Actual RUL (Clipped)')
    plt.ylabel('Predicted RUL')
    plt.title(f'Actual vs Predicted RUL ({identifier})')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/Predicted_vs_Actual_RUL_{identifier.replace(" ", "_")}.png", 
               bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    sample_units = random.sample(list(df[COLUMN_NAMES[Column.UnitNumber]].unique()), 5)
    for unit in sample_units:
        unit_data = df[df[COLUMN_NAMES[Column.UnitNumber]] == unit].copy()

        actual = unit_data[RUL_CLIPPED_COLUMN]
        pred = model.predict(unit_data[feature_cols])

        plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], actual, label=f'Unit {unit} - Actual', linestyle='-', marker='o')
        plt.plot(unit_data[COLUMN_NAMES[Column.TimeCycles]], pred, label=f'Unit {unit} - Predicted', linestyle='--')

    plt.xlabel('Cycle')
    plt.ylabel('RUL')
    plt.title('RUL Prediction over Cycles for Sample Engines')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/Sample_Engine_Predictions_{identifier.replace(" ", "_")}.png", 
               bbox_inches='tight', dpi=300)
    plt.close()

df_train = LoadData()
df_train = PrepareData(df_train)
df_train, feature_columns, sample_units = FeatureEngineering(df_train)
#df_train_split, df_test_split = TrainTestSplit(df_train)
#model, feature_cols, X_test, y_test, X_train, y_train = RandomForestModel(df_train_split, df_test_split, feature_columns)
#EvaluateModel(df_train_split, model, X_train, y_train, feature_cols, "Training Split")
#EvaluateModel(df_test_split, model, X_test, y_test, feature_cols, "Testing Split")
