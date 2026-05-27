# %%
import sys
import duckdb
import os
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# STEP 1 — Preprocessing

RANDOM_STATE = 202605
local_file_path = 'local_data_paris.csv'
ILE_DE_FRANCE = [75, 77, 78, 91, 92, 93, 94, 95]

if not os.path.exists(local_file_path):
    # Create a non-persistent connection (the database exists only while the connection is alive and disappears when it is closed)
    con = duckdb.connect(database=":memory:")
    df = con.sql(
        """
            SELECT * FROM read_parquet('https://minio.lab.sspcloud.fr/projet-funathon/2026/project1/data/1_input/transactions_EN.parquet')
        """).to_df()
    df.to_csv(local_file_path, index=False)
    df_paris = df[df.prop_loc_dep.isin([str(x) for x in ILE_DE_FRANCE])]
    df_paris = df_paris.copy()

else:
    # df = pd.read_csv(local_file_path, low_memory=False)
    df_paris = pd.read_csv(local_file_path)
    print(os.path.abspath(local_file_path))

# df.loc[:, 'prop_loc_dep'] = df.loc[:, 'prop_loc_dep'].astype(int)

# df.shape (9999635, 43)
# df.columns
"""['trans_date', 'trans_year', 'trans_month', 'price', 'prop_type',
       'prop_year_harm', 'prop_loc_dep', 'prop_loc_citycode', 'prop_loc_x',
       'prop_loc_y', 'dist_tosea', 'n_floors', 'n_bath', 'n_show', 'n_sink',
       'n_wc', 'n_mrooms', 'n_eatr', 'n_slr', 'n_kit8', 'n_kit9', 'n_washr',
       'n_ancrooms', 'n_rooms', 'farea', 'has_water', 'has_elec', 'stair',
       'has_gas', 'has_elevator', 'has_cheating', 'has_rchute',
       'has_mdrainage', 'nth_floor', 's_land_artif', 's_land_agri',
       's_land_nat', 'n_garage', 'n_pool', 'n_terrace', 'n_attic', 'n_basmt',
       'n_otherannex']"""
# transaction date: 1.1.2010 au 31.12.2024

# %%

df_paris['price_per_m2'] = df_paris.price / df_paris.farea
df_paris.drop('dist_tosea', axis=1, inplace=True)
df_paris.drop('has_elec', axis=1, inplace=True)

# %%
df_paris = df_paris[df_paris.price_per_m2 < 2.5e4]
df_paris = df_paris[df_paris.price_per_m2 > 99.9]
df_paris = df_paris.copy()

# %%
df_paris['price_per_m2_log'] = np.log(df_paris.price_per_m2.values)

x_mean = df_paris.prop_loc_x.mean()
y_mean = df_paris.prop_loc_y.mean()
#df_paris['dist_to_center'] = (df_paris.prop_loc_x - x_mean)**2+ \
#     (df_paris.prop_loc_y - y_mean)**2

# train — RMSE: 3011.52  |  MAE: 1801.93  |  R²: 0.3100
# test — RMSE: 3033.20  |  MAE: 1820.62  |  R²: 0.3007

y_mean, x_mean = 48.856789641455386, 2.352208093920579
df_paris['dist_to_center'] = (df_paris.prop_loc_x - x_mean)**2+ \
     (df_paris.prop_loc_y - y_mean)**2


# %%

import numpy as np
from sklearn.preprocessing import TargetEncoder
X = df_paris['has_cheating'].values.reshape((-1, 1))
y = df_paris.price_per_m2.values
enc_auto = TargetEncoder(smooth=0, cv=5)
X_trans = enc_auto.fit_transform(X, y)
df_paris['cheating_encoded'] = X_trans.ravel()
df_paris.groupby('cheating_encoded').price_per_m2.mean()
df_paris = df_paris.drop(columns=['has_cheating'])

# %% 
df_paris = df_paris.drop(columns=[
    "price", "prop_loc_dep", "prop_loc_citycode", 
])

df_paris["prop_type"] = pd.Categorical(
    df_paris["prop_type"].astype(str),
    categories=["1", "2"],
    ordered=False
).rename_categories({"1": "House", "2": "Flat"})

df_paris['prop_year_harm_10'] = (df_paris['prop_year_harm'] // 10)*10
df_paris['prop_year_harm_10'] = df_paris['prop_year_harm_10'].where(df_paris['prop_year_harm_10'] >= 1850, 1840)

# Dropping old column
df_paris = df_paris.drop(columns=["prop_year_harm"])


# %%
# STEP 2 — Train / test split, model fitting,
from sklearn.model_selection import train_test_split

# Split features / target
X = df_paris.drop(columns=["price_per_m2_log", "price_per_m2"])  # X must contain only the features we'll learn from
y = df_paris["price_per_m2"]  # target must be a dataframe with 1 column

# Split train / test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

# %%

import pandas as pd 
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline

RANDOM_STATE = 202605

def log_transform(y):
    return np.log10(y)

def inverse_log_transform(y):
    return 10 ** y

y_transformer = FunctionTransformer(
    func=log_transform,
    inverse_func=inverse_log_transform)

def date_to_days(X: pd.Series, ref_date: pd.Timestamp):
    # converts a date to a difference to ref_date :
    diff_dt = pd.to_datetime(X) - ref_date
    # Extract days part from datetime object
    diff_dt = diff_dt.dt.days
    # Transform it from a Pandas series to a Numpy nd array, used by scikit learn for input
    diff_dt = diff_dt.to_numpy().reshape(-1, 1)

    return diff_dt

date_transformer = FunctionTransformer(
    date_to_days,
    kw_args={"ref_date": pd.Timestamp('2010-01-01 00:00')}
    )

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["prop_type", "prop_year_harm_10"]),  # one-hot encoder on feature
        ("dat", date_transformer, "trans_date") # feature time since 01-01-2010
    ],
    remainder="passthrough"  # to keep features not transformed
)


# %%
from sklearn.ensemble import HistGradientBoostingRegressor

gb_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('GB', HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        early_stopping=True))
])

model = TransformedTargetRegressor(
    regressor=gb_pipeline,
    transformer=y_transformer
)

# %%

BEST_ITER = 500  # to automatically catch the best hyperparameter, set to : gs_step1.best_params_["regressor__GB__max_iter"]
BEST_LR = 0.25  # to automatically catch the best hyperparameter, set to : gs_step1.best_params_["regressor__GB__learning_rate"]
BEST_DEPTH = 20  # to automatically catch the best hyperparameter, set to : gs_step2.best_params_["regressor__GB__max_depth"]
BEST_MIN_LEAF = 75  # to automatically catch the best hyperparameter, set to : gs_step2.best_params_["regressor__GB__min_samples_leaf"]
BEST_L2 = 0


# %%
gb_final = HistGradientBoostingRegressor(
    max_iter=BEST_ITER,
    learning_rate=BEST_LR,
    max_depth=BEST_DEPTH,
    min_samples_leaf=BEST_MIN_LEAF,
    l2_regularization=BEST_L2,
    random_state=RANDOM_STATE,
)

# Wrap in the same pipeline / TransformedTargetRegressor as the RF section
gb_pipeline_best = Pipeline([
    ("preprocessor", preprocessor),  # same preprocessor as defined in the preprocessing section
    ("GB", gb_final),
])

gb_model_final = TransformedTargetRegressor(
    regressor=gb_pipeline_best,
    transformer=y_transformer  # same targettransformer as defined in preprocessing section
)

gb_model_final.fit(X_train, y_train)


def print_metrics(model, split, X, y):
    """
    Print metrics for trained model
    """
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{split} — RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R²: {r2:.4f}")


# %%

for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
    print_metrics(gb_model_final, split, X, y)

# %%
