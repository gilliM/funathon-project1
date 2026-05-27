# %%
import sys
import duckdb
import os
import pandas as pd

# STEP 1 — Preprocessing

RANDOM_STATE = 202605
local_file_path = 'local_data.csv'
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
    df_paris.to_csv('local_data_paris.csv', index=False)

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


def print_values(df):
    cols = df.columns
    for col in cols:
        print('column : {}'.format(col))
        print(df[col].drop_duplicates())
        print('na : {}'.format(df[col].isna().sum()))


# %%

df_paris['price_per_m2'] = df_paris.price / df_paris.farea

df_paris.drop('dist_tosea', axis=1, inplace=True)


# %%
df_paris = df_paris[df_paris.price_per_m2 < 2.5e4]
df_paris = df_paris[df_paris.price_per_m2 > 2000]

# %%
df_paris['price_per_m2_log'] = np.log(df_paris.price_per_m2.values)
df_paris.hist('price_per_m2', bins=100)

# %%

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
y = df_paris.price_per_m2
for ax, (data, label) in zip(axes, [(y[y <= 2000], "Y below 2000€ per sqm"), (y[y <= 500], "Y below 500€ per sqm")]):
    ax.hist(data, bins="auto", edgecolor="white", color="#334887", alpha=0.95)
    ax.set_title(label)
    ax.set_xlabel("Price per square meter")
    ax.set_ylabel("Number of transactions")

plt.tight_layout()
plt.show()

# %% 
plt.figure()
plt.plot(np.sin(np.linspace(-np.pi, np.pi, 1001)))
plt.show()

# %%
# ============================================
# STEP 2 — Train / test split, model fitting,
#
# ============================================

# YOUR CODE HERE

# %%
