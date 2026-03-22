import pandas as pd
import numpy as np

def small_categories_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    threshold = 0.95
    cols_to_drop = []
    for col in df.columns:
        max_freq = df[col].value_counts(normalize=True, dropna=False).max()
        if max_freq >= threshold and df[col].dtype in ['object', 'string']:
            cols_to_drop.append(col)
    print(f"Usunięto {len(cols_to_drop)} kolumn zdominowanych w >={threshold * 100}%:")
    print(cols_to_drop)
    for col in cols_to_drop:
        df.drop(col, axis=1, inplace=True)

    mapowanie_ocen = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Lack': 0}
    kolumny_do_zmiany = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                         'GarageQual', 'GarageCond']

    for kol in kolumny_do_zmiany:
        df[kol] = df[kol].map(mapowanie_ocen)

    freq = df['BldgType'].value_counts(normalize=True)
    df['BldgType'] = df['BldgType'].replace(freq[freq < 0.1].index.tolist(), 'Other')

    grades = {
        'Lack': 0,
        'No': 1,
        'Mn': 2,
        'Av': 3,
        'Gd': 4
    }
    df['BsmtExposure'] = df['BsmtExposure'].map(grades)

    grades = {
        'Lack': 0,
        'Unf': 1,
        'LwQ': 2,
        'Rec': 3,
        'BLQ': 4,
        'ALQ': 5,
        'GLQ': 6
    }
    df['BsmtFinType1'] = df['BsmtFinType1'].map(grades)
    df.rename(columns={'BsmtFinType1': 'BsmtFinType1Ovrl'}, inplace=True)

    df.drop('BsmtFinType2', axis=1, inplace=True)

    df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})

    grades = {
        'Artery': 'Noise',
        'RRAn': 'Noise',
        'RRAe': 'Noise',
        'Feedr': 'Noise',
        'RRNn': 'Noise',
        'RRNe': 'Noise',
        'Norm': 'Norm',
        'PosN': 'Pos',
        'PosA': 'Pos'
    }
    df['Condition1'] = df['Condition1'].map(grades)

    df['Electrical'] = df['Electrical'].apply(lambda x: 1 if x == 'SBrkr' else 0)
    df = df.rename(columns={'Electrical': 'IsStandardElectrical'})

    grades = {
        'NoFence': 0,
        'MnPrv': 1,
        'MnWw': 1,
        'GdWo': 2,
        'GdPrv': 2}
    df['Fence'] = df['Fence'].map(grades)
    df.rename(columns={'Fence': 'FenceOvrl'}, inplace=True)

    freq = df['Foundation'].value_counts(normalize=True)
    df['Foundation'] = df['Foundation'].replace(freq[freq < 0.11].index.tolist(), 'Other')

    df.drop('Functional', axis=1, inplace=True)

    grades = {
        'Lack': 0,
        'Unf': 1,
        'RFn': 2,
        'Fin': 3
    }
    df['GarageFinish'] = df['GarageFinish'].map(grades)

    freq = df['GarageType'].value_counts(normalize=True)
    df['GarageType'] = df['GarageType'].replace(freq[freq < 0.07].index.tolist(), 'Other')

    freq = df['HouseStyle'].value_counts(normalize=True)
    df['HouseStyle'] = df['HouseStyle'].replace(freq[freq < 0.11].index.tolist(), 'Other')

    df.drop('LandSlope', axis=1, inplace=True)

    df['LandContour'] = df['LandContour'].apply(lambda x: 1 if x == 'Lvl' else 0)
    df = df.rename(columns={'LandContour': 'IsFlat'})

    freq = df['LotConfig'].value_counts(normalize=True)
    df['LotConfig'] = df['LotConfig'].replace(freq[freq < 0.7].index.tolist(), 'Other')

    grades = {
        'IR3': 1,
        'IR2': 2,
        'IR1': 3,
        'Reg': 4
    }
    df['LotShape'] = df['LotShape'].map(grades)

    df['MSZoning'] = df['MSZoning'].replace(['FV', 'RH', "'C (all)'"], 'Other')

    df['MasVnrType'] = df['MasVnrType'].replace(['BrkFace', 'BrkCmn'], 'Brick')

    df['PavedDrive'] = df['PavedDrive'].map({'Y': 1, 'N': 0, 'P': 1})

    freq = df['RoofStyle'].value_counts(normalize=True)
    df['RoofStyle'] = df['RoofStyle'].replace(freq[freq < 0.1].index.tolist(), 'Other')

    freq = df['SaleCondition'].value_counts(normalize=True)
    df['SaleCondition'] = df['SaleCondition'].replace(freq[freq < 0.03].index.tolist(), 'Other')

    freq = df['SaleType'].value_counts(normalize=True)
    df['SaleType'] = df['SaleType'].replace(freq[freq < 0.025].index.tolist(), 'Other')

    not_numeric_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    columns_to_encode = [col for col in not_numeric_columns if
                            col not in ['Exterior1st', 'Exterior2nd', 'Neighborhood']]

    df = pd.get_dummies(df, columns=columns_to_encode, dtype=int)
    return df

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

def numerical_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ['3SsnPorch', 'BsmtFinSF2', 'Alley_Pave', 'Alley_Grvl', 'Alley_Pave', 'Condition1_Pos',
                    'KitchenAbvGr', 'LowQualFinSF', 'MasVnrType_Lack', 'RoofStyle_Other', 'SaleCondition_Other',
                    'SaleType_COD', 'SaleType_Other']
    df = df.drop(columns=cols_to_drop)

    add_binary_cols = ['PoolArea', 'EnclosedPorch', 'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'ScreenPorch', 'WoodDeckSF']
    for col in add_binary_cols:
        df[f'{col}_bin'] = (df[col] != 0).astype(int)

    cols_to_log = ['EnclosedPorch', 'MasVnrArea', 'OpenPorchSF', 'ScreenPorch', 'WoodDeckSF']
    for col in cols_to_log:
        df[f'{col}_log'] = np.log1p(df[col])

    df = df.drop(columns=add_binary_cols)

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
    df = df.sort_index(axis=1)

    return df