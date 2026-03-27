import pandas as pd
import numpy as np

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['Alley'] = df['Alley'].replace('?', 'NoAlley')

    kolumny_piwnica = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']
    for kol in kolumny_piwnica:
        df[kol] = df[kol].replace('?', np.nan)

    maska_brak_piwnicy = df['BsmtQual'].isna()
    for kol in kolumny_piwnica:
        df.loc[maska_brak_piwnicy, kol] = "Lack"
    for kol in ['BsmtExposure', 'BsmtFinType2']:
        najczestsza = df[kol].mode()[0]
        df[kol] = df[kol].fillna(najczestsza)
    kol = 'Electrical'
    zmiana = df[kol].mode()[0]
    df[kol] = df[kol].replace('?', zmiana)

    kol = 'Fence'
    zmiana = 'NoFence'
    df[kol] = df[kol].replace('?', zmiana)

    df.loc[df['Fireplaces'] == 0, 'FireplaceQu'] = 'Lack'

    garage_categorical = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in garage_categorical:
        df[col] = df[col].replace('?', 'Lack')

    df['GarageYrBlt'] = df['GarageYrBlt'].replace('?', 0)
    df['GarageYrBlt'] = pd.to_numeric(df['GarageYrBlt'])

    df['LotFrontage'] = df['LotFrontage'].replace('?', np.nan)
    df['LotFrontage'] = pd.to_numeric(df['LotFrontage'])
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    most_frequent_type = df['MasVnrType'].mode()[0]
    df.loc[(df['MasVnrType'].isnull()) & ((df['MasVnrArea'] == 0) | (df['MasVnrArea'] == 1)), 'MasVnrType'] = 'Lack'
    df['MasVnrType'] = df['MasVnrType'].replace('?', "Lack")
    df['MasVnrArea'] = df['MasVnrArea'].replace('?', 0)
    df['MasVnrArea'] = pd.to_numeric(df['MasVnrArea'])
    df['MasVnrType'] = df['MasVnrType'].fillna(most_frequent_type)

    kol = 'MiscFeature'
    zmiana = "Lack"
    df[kol] = df[kol].replace('?', zmiana)

    kol = 'PoolQC'
    zmiana = "Lack"
    df[kol] = df[kol].replace('?', zmiana)
    return df

def numerical_cleanup(df: pd.DataFrame, experiment_params: dict) -> pd.DataFrame:
    df['1stAnd2ndFlrSF'] = df['1stFlrSF'] + df['2ndFlrSF']
    df.drop('1stFlrSF', axis=1, inplace=True)
    df.drop('2ndFlrSF', axis=1, inplace=True)

    cols_with_years = ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd']

    reference_year = df['YrSold'].max()

    df['GarageAge'] = np.where(df['GarageYrBlt'] > 0, reference_year - df['GarageYrBlt'], 0)
    df['HasGarage'] = (df['GarageYrBlt'] > 0).astype(int)
    df['HouseAge'] = reference_year - df['YearBuilt']
    df['RemodAge'] = reference_year - df['YearRemodAdd']

    df = df.drop(columns=cols_with_years)

    threshold = experiment_params['drop_threshold']
    cols_to_drop = []
    for col in df.columns:
        max_freq = df[col].value_counts(normalize=True, dropna=False).max()
        if max_freq >= threshold:
            cols_to_drop.append(col)
    for col in cols_to_drop:
        df.drop(col, axis=1, inplace=True)

    target_cols = [col for col in df.columns if (df[col] == 0).mean() > 0.4 and df[col].nunique() > 11]
    for col in target_cols:
        df[f'{col}_bin'] = (df[col] != 0).astype(int)
    for col in target_cols:
        df[f'{col}_log'] = np.log1p(df[col])
    df = df.drop(columns=target_cols)

    return df

def encode_static_mappings(df: pd.DataFrame) -> pd.DataFrame:
    grades = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Lack': 0}
    columns_to_map = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                      'GarageQual', 'GarageCond']

    for col in columns_to_map:
        if col in df.columns:
            df[col] = df[col].map(grades)

    if 'BsmtExposure' in df.columns:
        grades = {'Lack': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
        df['BsmtExposure'] = df['BsmtExposure'].map(grades)

    if 'BsmtFinType1' in df.columns:
        grades = {'Lack': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
        df['BsmtFinType1'] = df['BsmtFinType1'].map(grades)
        df.rename(columns={'BsmtFinType1': 'BsmtFinType1Ovrl'}, inplace=True)

    if 'CentralAir' in df.columns:
        df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})

    if 'Condition1' in df.columns:
        grades = {'Artery': 'Noise', 'RRAn': 'Noise', 'RRAe': 'Noise', 'Feedr': 'Noise', 'RRNn': 'Noise',
                  'RRNe': 'Noise', 'Norm': 'Norm', 'PosN': 'Pos', 'PosA': 'Pos'}
        df['Condition1'] = df['Condition1'].map(grades)

    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].apply(lambda x: 1 if x == 'SBrkr' else 0)
        df.rename(columns={'Electrical': 'IsStandardElectrical'}, inplace=True)

    if 'Fence' in df.columns:
        grades = {'NoFence': 0, 'MnPrv': 1, 'MnWw': 1, 'GdWo': 2, 'GdPrv': 2}
        df['Fence'] = df['Fence'].map(grades)
        df.rename(columns={'Fence': 'FenceOvrl'}, inplace=True)

    if 'GarageFinish' in df.columns:
        grades = {'Lack': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        df['GarageFinish'] = df['GarageFinish'].map(grades)

    if 'LandContour' in df.columns:
        df['LandContour'] = df['LandContour'].apply(lambda x: 1 if x == 'Lvl' else 0)
        df.rename(columns={'LandContour': 'IsFlat'}, inplace=True)

    if 'LotShape' in df.columns:
        grades = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
        df['LotShape'] = df['LotShape'].map(grades)

    if 'PavedDrive' in df.columns:
        df['PavedDrive'] = df['PavedDrive'].map({'Y': 1, 'N': 0, 'P': 1})

    if 'MasVnrType' in df.columns:
        df['MasVnrType'] = df['MasVnrType'].replace(['BrkFace', 'BrkCmn'], 'Brick')
    return df

def group_rare_categories(df: pd.DataFrame, experiment_params: dict) -> pd.DataFrame:
    threshold = experiment_params['group_columns_threshold']

    not_numeric_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    columns_to_check = [col for col in not_numeric_columns if
                         col not in ['Exterior1st', 'Exterior2nd', 'Neighborhood']]

    for col in columns_to_check:
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index.tolist()

        if rare_categories:
            df[col] = df[col].replace(rare_categories, 'Other')
    return df

def drop_dominant_columns(df: pd.DataFrame, experiment_params: dict) -> pd.DataFrame:
    threshold = experiment_params['drop_columns_threshold']
    cols_to_drop = []
    for col in df.columns:
        max_freq = df[col].value_counts(normalize=True, dropna=False).max()
        if max_freq >= threshold and df[col].dtype in ['object', 'string']:
            cols_to_drop.append(col)
    for col in cols_to_drop:
        df.drop(col, axis=1, inplace=True)
    return df
