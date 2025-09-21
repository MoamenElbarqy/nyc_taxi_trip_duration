import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5)

def manhattan(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    return np.abs(lat2 - lat1) + np.abs(lon2 - lon1)


def haversine(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    R = 6371.0  # Radius of Earth in kilometers

    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences between coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # Central angle
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

class preprocessor:
    
    @staticmethod
    def add_manhattan_distance(df : pd.DataFrame):

        longitude_a = df['pickup_longitude'].to_numpy()
        latitude_a = df['pickup_latitude'].to_numpy()
        longitude_b = df['dropoff_longitude'].to_numpy()
        latitude_b = df['dropoff_latitude'].to_numpy()

        manhattan_distance = manhattan(latitude_a, longitude_a, latitude_b, longitude_b)

        df.insert(loc=len(df.columns) - 1, column='manhattan_distance',
        value=manhattan_distance)
        return 
    
    @staticmethod
    def add_haversine_distance(df : pd.DataFrame):
        longitude_a = df['pickup_longitude'].to_numpy()
        latitude_a = df['pickup_latitude'].to_numpy()
        longitude_b = df['dropoff_longitude'].to_numpy()
        latitude_b = df['dropoff_latitude'].to_numpy()

        great_circle_distance = haversine(latitude_a, longitude_a, latitude_b, longitude_b)
        
        df.insert(loc=len(df.columns) - 1, column='great_circule_distance',
                    value=great_circle_distance)
        
    @staticmethod  
    def drop_columns(df: pd.DataFrame, *columns_names: str):
        df.drop(columns=list(columns_names), inplace=True)

    @staticmethod
    def add_divided_date(df: pd.DataFrame):
        """
        This Function Seprate The DateTime With This Format 2016-06-08 07:36:19 to 
        dayofweek  month  dayofyear
            6        1       10
        """
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df.insert(loc=len(df.columns) - 1, column='dayofweek', value=df['pickup_datetime'].dt.dayofweek)
        df.insert(loc=len(df.columns) - 1, column='month', value=df['pickup_datetime'].dt.month)
        df.insert(loc=len(df.columns) - 1, column='hour', value=df['pickup_datetime'].dt.hour)
        df.insert(loc=len(df.columns) - 1, column='dayofyear', value=df['pickup_datetime'].dt.dayofyear)

    @staticmethod
    def one_hot_encoding_store_and_fwd_flag(df : pd.DataFrame):
        """
        We Can Use Pandas pd.get_dummies() but the order of the columns will be spoiled
        So We Can Make It Step By Step To Preserve The Columns Order
        """
        df['store_and_fwd_flag_Y'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
        df['store_and_fwd_flag_N'] = (df['store_and_fwd_flag'] == 'N').astype(int)
        idx = df.columns.get_loc("store_and_fwd_flag") + 1
        df.insert(idx, "store_and_fwd_flag_Y", df.pop("store_and_fwd_flag_Y"))
        df.insert(idx + 1, "store_and_fwd_flag_N", df.pop("store_and_fwd_flag_N"))

    @staticmethod
    def get_categorical_features(df: pd.DataFrame, *categorical_features_names) -> pd.DataFrame:
        return df[list(categorical_features_names)]
    
    @staticmethod
    def apply_standard_scaler(df : pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
        data = df.to_numpy()
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        return data, scaler
    
    @staticmethod
    def extract_target(df :pd.DataFrame) -> pd.DataFrame:
        target = df['trip_duration']
        preprocessor.drop_columns(df, 'trip_duration')
        return target.to_numpy().reshape(-1 ,1)
    
def preproccessing_data(df : pd.DataFrame) -> pd.DataFrame:

    preprocessor.add_haversine_distance(df)
    preprocessor.add_manhattan_distance(df)
    preprocessor.one_hot_encoding_store_and_fwd_flag(df)
    preprocessor.add_divided_date(df)
    return df
    
if __name__ == '__main__':
    df_train = pd.read_csv("/home/moamen/data_sets/1 project-nyc-taxi-trip-duration/split/train.csv")
    df_val = pd.read_csv("/home/moamen/data_sets/1 project-nyc-taxi-trip-duration/split/val.csv")
    df_test = pd.read_csv("/home/moamen/data_sets/1 project-nyc-taxi-trip-duration/split/test.csv")

    model = Ridge(fit_intercept=True, alpha=1)

    
    # Train 

    df_train = preproccessing_data(df_train)

    categorical_features = preprocessor.get_categorical_features(df_train,
                                                                'vendor_id',
                                                                'store_and_fwd_flag_Y',
                                                                'store_and_fwd_flag_N',
                                                                'dayofweek',
                                                                'month',
                                                                'hour',
                                                                'dayofyear').to_numpy()
    preprocessor.drop_columns(df_train,
                            'id',
                            'pickup_datetime',
                            'vendor_id',
                            'store_and_fwd_flag',
                            'store_and_fwd_flag_Y',
                            'store_and_fwd_flag_N',
                            'dayofweek',
                            'month',
                            'hour',
                            'dayofyear')
    
    t_train = np.log1p(preprocessor.extract_target(df_train))

    scaled_continuous_features, scaler = preprocessor.apply_standard_scaler(df_train)

    x_train = np.hstack([scaled_continuous_features, categorical_features])
    
    model.fit(x_train, t_train)

    train_pred = model.predict(x_train)

    train_error = r2_score(t_train, train_pred)
    print(f"Train Error  {train_error}")

    # validation 
    df_val = preproccessing_data(df_val)
    categorical_features_val = preprocessor.get_categorical_features(df_val, 'vendor_id', 'store_and_fwd_flag_Y', 'store_and_fwd_flag_N', 'dayofweek', 'month', 'hour', 'dayofyear').to_numpy()
    preprocessor.drop_columns(df_val, 'id', 'pickup_datetime', 'vendor_id', 'store_and_fwd_flag', 'store_and_fwd_flag_Y', 'store_and_fwd_flag_N', 'dayofweek', 'month', 'hour', 'dayofyear')
    t_val = preprocessor.extract_target(df_val)
    scaled_continuous_features_val = scaler.transform(df_val)  # Use the same scaler from training
    x_val = np.hstack([scaled_continuous_features_val, categorical_features_val])
    val_pred = model.predict(x_val)
    val_error = r2_score(np.log1p(t_val), val_pred)  # Assuming log1p on target
    print(f"Validation Error {val_error}")

# train error 0.32406262386227747
# Validation Error 0.35912651935515727