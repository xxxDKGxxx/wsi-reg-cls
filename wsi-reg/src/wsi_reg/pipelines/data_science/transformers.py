import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_cols=None):
        self.exclude_cols = exclude_cols
        self.encoder_ = None
        self.cols_to_encode_ = []

    def fit(self, X, y=None):
        exclude = self.exclude_cols if self.exclude_cols is not None else []
        all_obj_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
        self.cols_to_encode_ = [c for c in all_obj_cols if c not in exclude]

        if self.cols_to_encode_:
            self.encoder_ = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            self.encoder_.fit(X[self.cols_to_encode_])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if not self.cols_to_encode_ or self.encoder_ is None:
            return X_transformed

        encoded_arrays = self.encoder_.transform(X_transformed[self.cols_to_encode_])
        feature_names = self.encoder_.get_feature_names_out(self.cols_to_encode_)

        encoded_df = pd.DataFrame(encoded_arrays, columns=feature_names, index=X_transformed.index)

        X_transformed = X_transformed.drop(columns=self.cols_to_encode_)
        X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

        return X_transformed

class DataFrameMissingValuesImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bsmt_exposure_mode_ = None
        self.bsmt_fintype2_mode_ = None
        self.electrical_mode_ = None
        self.lot_frontage_medians_ = {}
        self.mas_vnr_type_mode_ = None

    def fit(self, X, y=None):
        self.bsmt_exposure_mode_ = X['BsmtExposure'].mode()[0] if not X['BsmtExposure'].mode().empty else 'No'
        self.bsmt_fintype2_mode_ = X['BsmtFinType2'].mode()[0] if not X['BsmtFinType2'].mode().empty else 'Unf'
        self.electrical_mode_ = X['Electrical'].mode()[0] if not X['Electrical'].mode().empty else 'SBrkr'

        temp_lot = X['LotFrontage'].replace('?', np.nan).astype(float)
        self.lot_frontage_medians_ = temp_lot.groupby(X['Neighborhood']).median().to_dict()
        self.global_lot_median_ = temp_lot.median()

        self.mas_vnr_type_mode_ = X['MasVnrType'].mode()[0] if not X['MasVnrType'].mode().empty else 'None'
        return self

    def transform(self, X):
        X_transformed = X.copy()

        X_transformed['Alley'] = X_transformed['Alley'].replace('?', 'NoAlley')

        kolumny_piwnica = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual']
        for kol in kolumny_piwnica:
            X_transformed[kol] = X_transformed[kol].replace('?', np.nan)

        maska_brak_piwnicy = X_transformed['BsmtQual'].isna()
        for kol in kolumny_piwnica:
            X_transformed.loc[maska_brak_piwnicy, kol] = "Lack"

        X_transformed['BsmtExposure'] = X_transformed['BsmtExposure'].fillna(self.bsmt_exposure_mode_)
        X_transformed['BsmtFinType2'] = X_transformed['BsmtFinType2'].fillna(self.bsmt_fintype2_mode_)

        X_transformed['Electrical'] = X_transformed['Electrical'].replace('?', self.electrical_mode_)
        X_transformed['Fence'] = X_transformed['Fence'].replace('?', 'NoFence')
        X_transformed.loc[X_transformed['Fireplaces'] == 0, 'FireplaceQu'] = 'Lack'

        garage_categorical = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        for col in garage_categorical:
            X_transformed[col] = X_transformed[col].replace('?', 'Lack')

        X_transformed['GarageYrBlt'] = pd.to_numeric(X_transformed['GarageYrBlt'].replace('?', 0))

        X_transformed['LotFrontage'] = pd.to_numeric(X_transformed['LotFrontage'].replace('?', np.nan))
        X_transformed['LotFrontage'] = X_transformed.apply(
            lambda row: self.lot_frontage_medians_.get(row['Neighborhood'], self.global_lot_median_)
            if pd.isna(row['LotFrontage']) else row['LotFrontage'],
            axis=1
        )

        X_transformed.loc[(X_transformed['MasVnrType'].isnull()) & (
                    (X_transformed['MasVnrArea'] == 0) | (X_transformed['MasVnrArea'] == 1)), 'MasVnrType'] = 'Lack'
        X_transformed['MasVnrType'] = X_transformed['MasVnrType'].replace('?', "Lack")
        X_transformed['MasVnrArea'] = pd.to_numeric(X_transformed['MasVnrArea'].replace('?', 0))
        X_transformed['MasVnrType'] = X_transformed['MasVnrType'].fillna(self.mas_vnr_type_mode_)

        X_transformed['MiscFeature'] = X_transformed['MiscFeature'].replace('?', "Lack")
        X_transformed['PoolQC'] = X_transformed['PoolQC'].replace('?', "Lack")

        return X_transformed


class DataFrameNumericalEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_threshold=0.98):
        self.drop_threshold = drop_threshold
        self.reference_year_ = None
        self.cols_to_drop_ = []
        self.target_cols_ = []

    def fit(self, X, y=None):
        self.reference_year_ = X['YrSold'].max()

        X_temp = X.copy()
        X_temp['1stAnd2ndFlrSF'] = X_temp['1stFlrSF'] + X_temp['2ndFlrSF']
        X_temp.drop(columns=['1stFlrSF', '2ndFlrSF'], inplace=True, errors='ignore')
        X_temp['GarageYrBlt'] = pd.to_numeric(X_temp['GarageYrBlt'].replace('?', 0))
        X_temp['GarageAge'] = np.where(X_temp['GarageYrBlt'] > 0, self.reference_year_ - X_temp['GarageYrBlt'], 0)
        X_temp['HasGarage'] = (X_temp['GarageYrBlt'] > 0).astype(int)
        X_temp['HouseAge'] = self.reference_year_ - X_temp['YearBuilt']
        X_temp['RemodAge'] = self.reference_year_ - X_temp['YearRemodAdd']
        X_temp.drop(columns=['GarageYrBlt', 'YearBuilt', 'YearRemodAdd'], inplace=True, errors='ignore')

        self.cols_to_drop_ = []
        for col in X_temp.columns:
            if not X_temp[col].empty:
                max_freq = X_temp[col].value_counts(normalize=True, dropna=False).max()
                if max_freq >= self.drop_threshold:
                    self.cols_to_drop_.append(col)

        X_temp.drop(columns=self.cols_to_drop_, inplace=True, errors='ignore')

        self.target_cols_ = [col for col in X_temp.columns if
                             pd.api.types.is_numeric_dtype(X_temp[col]) and (X_temp[col] == 0).mean() > 0.4 and X_temp[
                                 col].nunique() > 11]

        return self

    def transform(self, X):
        X_transformed = X.copy()

        X_transformed['1stAnd2ndFlrSF'] = X_transformed['1stFlrSF'] + X_transformed['2ndFlrSF']
        X_transformed.drop(columns=['1stFlrSF', '2ndFlrSF'], inplace=True, errors='ignore')

        cols_with_years = ['GarageYrBlt', 'YearBuilt', 'YearRemodAdd']

        X_transformed['GarageYrBlt'] = pd.to_numeric(X_transformed['GarageYrBlt'].replace('?', 0))
        X_transformed['GarageAge'] = np.where(X_transformed['GarageYrBlt'] > 0,
                                              self.reference_year_ - X_transformed['GarageYrBlt'], 0)
        X_transformed['HasGarage'] = (X_transformed['GarageYrBlt'] > 0).astype(int)
        X_transformed['HouseAge'] = self.reference_year_ - X_transformed['YearBuilt']
        X_transformed['RemodAge'] = self.reference_year_ - X_transformed['YearRemodAdd']

        X_transformed.drop(columns=cols_with_years, inplace=True, errors='ignore')

        cols_to_drop = [c for c in self.cols_to_drop_ if c in X_transformed.columns]
        X_transformed.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        target_cols = [c for c in self.target_cols_ if c in X_transformed.columns]
        for col in target_cols:
            X_transformed[f'{col}_bin'] = (X_transformed[col] != 0).astype(int)
            X_transformed[f'{col}_log'] = np.log1p(X_transformed[col])
        X_transformed.drop(columns=target_cols, inplace=True, errors='ignore')

        return X_transformed.sort_index(axis=1)


class DataFrameStaticMappingsEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        grades_qual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Lack': 0}
        columns_to_map = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                          'GarageQual', 'GarageCond']
        for col in columns_to_map:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].map(grades_qual)

        if 'BsmtExposure' in X_transformed.columns:
            X_transformed['BsmtExposure'] = X_transformed['BsmtExposure'].map(
                {'Lack': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
        if 'BsmtFinType1' in X_transformed.columns:
            X_transformed['BsmtFinType1'] = X_transformed['BsmtFinType1'].map(
                {'Lack': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6})
            X_transformed.rename(columns={'BsmtFinType1': 'BsmtFinType1Ovrl'}, inplace=True)
        if 'CentralAir' in X_transformed.columns:
            X_transformed['CentralAir'] = X_transformed['CentralAir'].map({'Y': 1, 'N': 0})
        if 'Condition1' in X_transformed.columns:
            X_transformed['Condition1'] = X_transformed['Condition1'].map(
                {'Artery': 'Noise', 'RRAn': 'Noise', 'RRAe': 'Noise', 'Feedr': 'Noise', 'RRNn': 'Noise',
                 'RRNe': 'Noise', 'Norm': 'Norm', 'PosN': 'Pos', 'PosA': 'Pos'})
        if 'Electrical' in X_transformed.columns:
            X_transformed['Electrical'] = X_transformed['Electrical'].apply(lambda x: 1 if x == 'SBrkr' else 0)
            X_transformed.rename(columns={'Electrical': 'IsStandardElectrical'}, inplace=True)
        if 'Fence' in X_transformed.columns:
            X_transformed['Fence'] = X_transformed['Fence'].map(
                {'NoFence': 0, 'MnPrv': 1, 'MnWw': 1, 'GdWo': 2, 'GdPrv': 2})
            X_transformed.rename(columns={'Fence': 'FenceOvrl'}, inplace=True)
        if 'GarageFinish' in X_transformed.columns:
            X_transformed['GarageFinish'] = X_transformed['GarageFinish'].map({'Lack': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
        if 'LandContour' in X_transformed.columns:
            X_transformed['LandContour'] = X_transformed['LandContour'].apply(lambda x: 1 if x == 'Lvl' else 0)
            X_transformed.rename(columns={'LandContour': 'IsFlat'}, inplace=True)
        if 'LotShape' in X_transformed.columns:
            X_transformed['LotShape'] = X_transformed['LotShape'].map({'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4})
        if 'PavedDrive' in X_transformed.columns:
            X_transformed['PavedDrive'] = X_transformed['PavedDrive'].map({'Y': 1, 'N': 0, 'P': 1})
        if 'MasVnrType' in X_transformed.columns:
            X_transformed['MasVnrType'] = X_transformed['MasVnrType'].replace(['BrkFace', 'BrkCmn'], 'Brick')

        return X_transformed


class DataFrameDominantTextColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, drop_threshold=0.98):
        self.drop_threshold = drop_threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        self.cols_to_drop_ = []
        for col in X.select_dtypes(include=['object', 'string']).columns:
            if not X[col].empty:
                max_freq = X[col].value_counts(normalize=True, dropna=False).max()
                if max_freq >= self.drop_threshold:
                    self.cols_to_drop_.append(col)
        return self

    def transform(self, X):
        X_transformed = X.copy()
        cols_to_drop = [c for c in self.cols_to_drop_ if c in X_transformed.columns]
        X_transformed.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        return X_transformed


class DataFrameRareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        self.frequent_categories_ = {}
        categorical_cols = X.select_dtypes(include=['object', 'string']).columns
        columns_to_check = [col for col in categorical_cols if
                            col not in ['Exterior1st', 'Exterior2nd', 'Neighborhood']]

        for col in columns_to_check:
            freq = X[col].value_counts(normalize=True)
            self.frequent_categories_[col] = freq[freq >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, freq_cats in self.frequent_categories_.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].apply(lambda val: val if val in freq_cats else 'Other')
        return X_transformed


class DataFrameTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str], smoothing: float = 1.0):
        self.cols = cols
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DataFrameTargetEncoder":
        self.global_mean_ = float(y.mean())
        self.encoding_maps_ = {}

        for col in self.cols:
            stats = (
                y.groupby(X[col])
                .agg(["mean", "count"])
                .rename(columns={"mean": "cat_mean", "count": "n"})
            )
            smoother = stats["n"] / (stats["n"] + self.smoothing)
            stats["encoded"] = smoother * stats["cat_mean"] + (1 - smoother) * self.global_mean_
            self.encoding_maps_[col] = stats["encoded"].to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols:
            X_out[col] = X_out[col].map(self.encoding_maps_[col]).fillna(self.global_mean_)
        return X_out


class DataFrameCategoryConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_out = X.copy()
        for col in X_out.select_dtypes(include='object').columns:
            X_out[col] = X_out[col].astype('category')
        return X_out

class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_ = StandardScaler()
        self.num_cols_ = []

    def fit(self, X, y=None):
        self.num_cols_ = X.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        if self.num_cols_:
            self.scaler_.fit(X[self.num_cols_])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if self.num_cols_ and hasattr(self.scaler_, 'scale_'):
            X_transformed[self.num_cols_] = self.scaler_.transform(X_transformed[self.num_cols_])
        return X_transformed
