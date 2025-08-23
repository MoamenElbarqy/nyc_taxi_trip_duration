import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
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

def preproccessing_data(df : pd.DataFrame) -> np.ndarray:
    longitude_a = df['pickup_longitude'].to_numpy()
    latitude_a = df['pickup_latitude'].to_numpy()
    longitude_b = df['dropoff_longitude'].to_numpy()
    latitude_b = df['dropoff_latitude'].to_numpy()

    great_circle_distance = pd.DataFrame(haversine(latitude_a, longitude_a, latitude_b, longitude_b))
    manhattan_distance = pd.DataFrame(manhattan(latitude_a, longitude_a, latitude_b, longitude_b))

    df.insert(loc=len(df.columns) - 1, column='great_circule_distance',
                    value=great_circle_distance)
    df.insert(loc=len(df.columns) - 1, column='manhattan_distance',
              value=manhattan_distance)


    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df['store_and_fwd_flag_Y'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    df['store_and_fwd_flag_N'] = (df['store_and_fwd_flag'] == 'N').astype(int)

    idx = df.columns.get_loc("store_and_fwd_flag") + 1
    df.insert(idx, "store_and_fwd_flag_Y", df.pop("store_and_fwd_flag_Y"))
    df.insert(idx + 1, "store_and_fwd_flag_N", df.pop("store_and_fwd_flag_N"))

    df.drop(columns=["store_and_fwd_flag"], inplace=True)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])


    df.insert(loc=len(df.columns) - 1, column='dayofweek', value=df['pickup_datetime'].dt.dayofweek)
    df.insert(loc=len(df.columns) - 1, column='month', value=df['pickup_datetime'].dt.month)
    df.insert(loc=len(df.columns) - 1, column='hour', value=df['pickup_datetime'].dt.hour)
    df.insert(loc=len(df.columns) - 1, column='dayofyear', value=df['pickup_datetime'].dt.dayofyear)


    df.drop(columns=['id', 'pickup_datetime', 'hour'], inplace=True)
    print(df)
    return df.to_numpy()

if __name__ == '__main__':
    df_train = pd.read_csv("/home/moamen/data_sets/1 project-nyc-taxi-trip-duration/split/train.csv")
    df_val = pd.read_csv("/home/moamen/data_sets/1 project-nyc-taxi-trip-duration/split/val.csv")
    df_test = pd.read_csv("/home/moamen/data_sets/1 project-nyc-taxi-trip-duration/split/test.csv")

    model = Ridge(fit_intercept=True, alpha=1)

    # --- Train ---
    data_train = preproccessing_data(df_train)
    scaler = MinMaxScaler()

    x_train = data_train[:, :-1]
    y_train = np.log1p(data_train[:, -1].reshape(-1, 1))

    x_train_scaled = scaler.fit_transform(x_train)
    model.fit(x_train_scaled, y_train)

    train_pred = model.predict(x_train_scaled)
    train_error = r2_score(y_train, train_pred)
    print(f"Train Error  {train_error}")

    # --- Validation ---
    data_val = preproccessing_data(df_val)

    x_val = data_val[:, :-1]
    y_val = np.log1p(data_val[:, -1].reshape(-1, 1))

    # important: use transform not fit_transform
    x_val_scaled = scaler.transform(x_val)

    val_pred = model.predict(x_val_scaled)
    val_error = r2_score(y_val, val_pred)
    print(f"Validation Error  {val_error}")


    # without log target Train Error  0.005364337184089907
    # with log target Train Error  0.05518502575128714
    # when keeping outliers and didn't remove trips more than 1 day Train Error  0.32163572097904125

