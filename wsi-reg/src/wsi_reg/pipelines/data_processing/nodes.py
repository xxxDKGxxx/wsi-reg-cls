import pandas as pd
import numpy as np

def small_categories_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df['LotFrontage'] = pd.to_numeric(df['LotFrontage'], errors='coerce').astype('Int64')
    df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'], errors='coerce').astype('Int64')
    df['GarageYrBlt'] = pd.to_numeric(df['GarageYrBlt'], errors='coerce').astype('Int64')

    cat_features_s = []
    cat_features_b = []
    num_features = []

    for col in df.columns:
        if (df[col].dtype == 'object' and df[col].nunique() < 10):
            cat_features_s.append(col)
        elif df[col].dtype == 'object' and df[col].nunique() >= 10:
            cat_features_b.append(col)
        else:
            num_features.append(col)

    df_cat_s = df[cat_features_s]
    df_cat_b = df[cat_features_b]
    df_num = df[num_features]

    #df_cat_s = df_cat_s.fillna('?')

    mapowanie_ocen = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, "None": 0}
    kolumny_do_zmiany = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                         'GarageQual', 'GarageCond']

    for kol in kolumny_do_zmiany:
        df_cat_s[kol] = df_cat_s[kol].map(mapowanie_ocen).astype('Int64')
    #df_cat_s = df_cat_s.fillna('?')

    df_cat_s['MSZoning'] = df_cat_s['MSZoning'].replace(['FV', 'RH', "'C (all)'"], 'Other')

    df_cat_s.drop('Street', axis=1, inplace=True)

    df_cat_s['Alley'] = df_cat_s['Alley'].replace('?', 'NoAlley')

    df_cat_s['LotShape'] = df_cat_s['LotShape'].replace(['IR2', 'IR3'], 'Other')

    df_cat_s['LandContour'] = df_cat_s['LandContour'].replace(['Bnk', 'Low', 'HLS'], 'Other')

    df_cat_s.drop('Utilities', axis=1, inplace=True)

    df_cat_s['LotConfig'] = df_cat_s['LotConfig'].replace(['FR2', 'CulDSac', 'FR3'], 'Other')

    df_cat_s['LandSlope'] = df_cat_s['LandSlope'].replace(['Sev', 'Mod'], 'Other')

    df_cat_s['Condition1'] = df_cat_s['Condition1'].replace(
        ['Feedr', 'PosN', 'Artery', 'RRAn', 'RRAe', 'RRNn', 'PosA', 'RRNe'], 'Other')

    df_cat_s.drop('Condition2', axis=1, inplace=True)

    freq = df_cat_s['BldgType'].value_counts(normalize=True)
    df_cat_s['BldgType'] = df_cat_s['BldgType'].replace(freq[freq < 0.1].index.tolist(), 'Other')

    freq = df_cat_s['HouseStyle'].value_counts(normalize=True)
    df_cat_s['HouseStyle'] = df_cat_s['HouseStyle'].replace(freq[freq < 0.11].index.tolist(), 'Other')

    freq = df_cat_s['RoofStyle'].value_counts(normalize=True)
    df_cat_s['RoofStyle'] = df_cat_s['RoofStyle'].replace(freq[freq < 0.1].index.tolist(), 'Other')

    df_cat_s.drop('RoofMatl', axis=1, inplace=True)

    df_cat_s['MasVnrType'] = df_cat_s['MasVnrType'].replace(['BrkFace', 'BrkCmn'], 'Brick')

    freq = df_cat_s['Foundation'].value_counts(normalize=True)
    df_cat_s['Foundation'] = df_cat_s['Foundation'].replace(freq[freq < 0.11].index.tolist(), 'Other')

    freq = df_cat_s['BsmtExposure'].value_counts(normalize=True)
    df_cat_s['BsmtExposure'] = df_cat_s['BsmtExposure'].replace(freq[freq < 0.08].index.tolist(), 'Other')

    freq = df_cat_s['BsmtFinType1'].value_counts(normalize=True)
    df_cat_s['BsmtFinType1'] = df_cat_s['BsmtFinType1'].replace(freq[freq < 0.1].index.tolist(), 'Other')

    df_cat_s.drop('BsmtFinType2', axis=1, inplace=True)

    df_cat_s.drop('Heating', axis=1, inplace=True)

    df_cat_s['CentralAir'] = df_cat_s['CentralAir'].map({'Y': 1, 'N': 0})

    freq = df_cat_s['Electrical'].value_counts(normalize=True)
    df_cat_s['Electrical'] = df_cat_s['Electrical'].replace(freq[freq < 0.07].index.tolist(), 'Other')

    df_cat_s.drop('Functional', axis=1, inplace=True)

    freq = df_cat_s['GarageType'].value_counts(normalize=True)
    df_cat_s['GarageType'] = df_cat_s['GarageType'].replace(freq[freq < 0.06].index.tolist(), 'Other')

    df_cat_s['PavedDrive'] = df_cat_s['PavedDrive'].map({'Y': 1, 'N': 0, 'P': 1})

    df_cat_s.drop('PoolQC', axis=1, inplace=True)

    oceny_ogrodzenia = {
        '?': 0,
        'None': 0,
        'MnPrv': 1,
        'MnWw': 1,
        'GdWo': 2,
        'GdPrv': 2}
    df_cat_s['Fence'] = df_cat_s['Fence'].replace(oceny_ogrodzenia)
    df_cat_s.rename(columns={'Fence': 'FenceOvrl'}, inplace=True)

    df_cat_s['Has_MiscFeature'] = 1
    df_cat_s.loc[df_cat_s['MiscFeature'] == '?', 'Has_MiscFeature'] = 0
    df_cat_s.drop('MiscFeature', axis=1, inplace=True)

    freq = df_cat_s['SaleType'].value_counts(normalize=True)
    df_cat_s['SaleType'] = df_cat_s['SaleType'].replace(freq[freq < 0.03].index.tolist(), 'Other')

    freq = df_cat_s['SaleCondition'].value_counts(normalize=True)
    df_cat_s['SaleCondition'] = df_cat_s['SaleCondition'].replace(freq[freq < 0.03].index.tolist(), 'Other')

    df_cleaned = pd.concat([df_num, df_cat_s, df_cat_b], axis=1)
    return df_cleaned

def small_categories_enc(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if (df[col].dtype == 'object' and df[col].nunique() < 10):
            df = pd.get_dummies(df, columns=[col], drop_first=False)
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['Alley'] = df['Alley'].replace('?', 'NoAlley')

    kolumny_piwnica = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']
    for kol in kolumny_piwnica:
        df[kol] = df[kol].replace('?', np.nan)

    maska_brak_piwnicy = df['BsmtQual'].isna()
    for kol in kolumny_piwnica:
        df.loc[maska_brak_piwnicy, kol] = 0
    for kol in ['BsmtExposure', 'BsmtFinType2']:
        najczestsza = df[kol].mode()[0]
        df[kol] = df[kol].fillna(najczestsza)
    kol = 'Electrical'
    zmiana = df[kol].mode()[0]
    df[kol] = df[kol].replace('?', zmiana)

    kol = 'Fence'
    zmiana = 'NoFence'
    df[kol] = df[kol].replace('?', zmiana)

    df.loc[df['Fireplaces'] == 0, 'FireplaceQu'] = 'None'

    garage_categorical = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_categorical:
        df[col] = df[col].replace('?', 'None')

    df['GarageYrBlt'] = df['GarageYrBlt'].replace('?', 0)
    df['GarageYrBlt'] = pd.to_numeric(df['GarageYrBlt'])

    df['LotFrontage'] = df['LotFrontage'].replace('?', np.nan)
    df['LotFrontage'] = pd.to_numeric(df['LotFrontage'])
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    most_frequent_type = df['MasVnrType'].mode()[0]
    df.loc[(df['MasVnrType'].isnull()) & ((df['MasVnrArea'] == 0) | (df['MasVnrArea'] == 1)), 'MasVnrType'] = 'None'
    df['MasVnrType'] = df['MasVnrType'].replace('?', "None")
    df['MasVnrArea'] = df['MasVnrArea'].replace('?', 0)
    df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'])
    df['MasVnrType'] = df['MasVnrType'].fillna(most_frequent_type)

    kol = 'MiscFeature'
    zmiana = 'None'
    df[kol] = df[kol].replace('?', zmiana)

    kol = 'PoolQC'
    zmiana = 'None'
    df[kol] = df[kol].replace('?', zmiana)
    return df