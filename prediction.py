# coding: utf-8

"""
Bikes availability prediction (i.e. probability) using xgboost.
"""


import logging
import daiquiri

import numpy as np
import pandas as pd
from dateutil import parser
from workalendar.europe import France
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from tslearn.piecewise import PiecewiseAggregateApproximation, OneD_SymbolicAggregateApproximation
import xgboost as xgb

# French Calendar
cal = France()


SEED = 1337
np.random.seed(SEED)

daiquiri.setup(logging.INFO)
logger = daiquiri.getLogger("prediction")


def datareader(fpath):
    """Read a CSV file ane return a DataFrame
    """
    logger.info("read the file '%s'", fpath)
    coldate = 'last_update'
    return pd.read_csv(fpath, parse_dates=[coldate])


def complete_data(df):
    """Add some columns

    - day of the week
    - hour of the day
    - minute (10 by 10)
    """
    logger.info("complete some data")
    # def group_minute(value):
    #     if value <= 10:
    #         return 0
    #     if value <= 20:
    #         return 10
    #     if value <= 30:
    #         return 20
    #     if value <= 40:
    #         return 30
    #     if value <= 50:
    #         return 40
    #     return 50
    df = df.copy()
    df['day'] = df['ts'].apply(lambda x: x.weekday())
    df['hour'] = df['ts'].apply(lambda x: x.hour)
    # minute = df['ts'].apply(lambda x: x.minute)
    # df['minute'] = minute.apply(group_minute)
    df['minute'] = df['ts'].apply(lambda x: x.minute)
    return df


def cleanup(df):
    """Clean up

    - #keep OPEN station
    - drop duplicates
    - rename some columns
    - drop some columns
    - drop lines when stands == bikes == 0
    """
    logger.info("cleanup processing")
    columns_to_drop = ['availability', 
                        #'status', Use it for news features
                       'bike_stands', 'availabilitycode']
    df = (df.copy()
          #.query("status == 'OPEN'") Taking all data (OPEN & CLOSE)
          .drop(columns_to_drop, axis=1)
          .drop_duplicates()
          .rename_axis({"available_bike_stands": "stands",
                        "available_bikes": "bikes",
                        "last_update": "ts",
                        "number": "station"}, axis=1)
          .query("stands > 0 and bikes > 0")) # or 'Gris' availability value...
    return df


def availability(df, threshold):
    """Set an 'availability' column according to a threshold

    if the number of bikes is less than `threshold`, the availability (of bikes,
    not stands) is low.
    """
    logger.info("set the availability level")
    df = df.copy()
    key = 'availability'
    df[key] = 'medium'
    low_mask = df['bikes'] < threshold
    high_mask = np.logical_and(np.logical_not(low_mask),
                               df['stands'] < threshold)
    df.loc[low_mask, key] = 'low'
    df.loc[high_mask, key] = 'high'
    return df


def bikes_probability(df):
    logger.info("bikes probability")
    df['probability'] = df['bikes'] / (df['bikes'] + df['stands'])
    return df


def extract_bonus_by_station(df):
    """Return a series with station id and bonus oui/non

    turn the french yes/no into 1/0
    """
    logger.info("extract the bonus for each station")
    result =  (df.groupby(["station", "bonus"])["bikes"]
               .count()
               .reset_index())
    result['bonus'] = result['bonus'].apply(lambda x: 1 if x == 'Oui' else 0)
    return result[["station", "bonus"]].set_index("station")


def time_resampling(df, freq="10T"):
    """Normalize the timeseries
    """
    logger.info("Time resampling for each station by '%s'", freq)

    # Transform `status` Features in nemerial in 'is_open'

    df['is_open'] = 0
    df.loc[df['status'] == "OPEN", 'is_open'] = 1

    df = (df.groupby("station")
          .resample(freq, on="ts")[["ts", "bikes", "stands", "is_open"]]
          .mean()
          .bfill())
    return df.reset_index()


def prepare_data_for_training(df, date, freq='1H', start=None, periods=1,
                              observation='availability', how='train_test_split'):
    """Prepare data for training

    date: datetime / Timestamp
        date for the prediction
    freq: str
        the delay between the latest available data and the prediction. e.g. one hour
    start: Timestamp
        start of the history data (for training)
    periods: int
        number of predictions
    how : string
        - train_test_split : Performe a train_test_split returnning
                             train_X, train_Y, test_X, test_Y
        - None : Return X, y

    Returns 4 DataFrames: two for training, two for testing
    """
    logger.info("prepare data for training")
    logger.info("New version 4")
    logger.info("Get summer holiday features")
    df = get_summer_holiday(df)
    logger.info("Get public holiday features")
    df = get_public_holiday(df, count_day=5)
    logger.info("Get cluster station features")
    df = cluster_station_lyon(df)
    logger.info("Get Geo cluster station features")
    df = cluster_station_geo_lyon(df)
    # logger.info("Get ratio station open by time")
    # df = get_statio_ratio_open_by_time(df)
    logger.info("Get ratio station geo cluster open by time")
    df = get_statio_cluster_geo_ratio_open_by_time(df)
    logger.info("sort values (station, ts)")
    data = df.sort_values(['station', 'ts']).set_index(["ts", "station"])
    logger.info("compute the future availability at '%s'", freq)
    label = data[observation].copy()
    label.name = "future"
    label = (label.reset_index(level=1)
             .shift(-1, freq=freq)
             .reset_index()
             .set_index(["ts", "station"]))
    logger.info("merge data with the future availability")
    result = data.merge(label, left_index=True, right_index=True)
    logger.info("availability label as values")
    if observation == 'availability':
        result[observation] = result[observation].replace({"low": 0, "medium": 1, "high": 2})
        result['future'] = result['future'].replace({"low": 0, "medium": 1, "high": 2})
    result.reset_index(level=1, inplace=True)
    if start is not None:
        result = result[result.index >= start]

    logger.info("Create shift features")
    result = create_shift_features(result, features_name='bikes_shift_'+str(freq)+'min', feature_to_shift='bikes', 
                                features_grp='station', nb_shift=periods)
    logger.info("Create cumulative trending features")
    result = create_cumul_trend_features(result, features_name='bikes_shift_'+str(freq)+'min')
    logger.info("Create recenlty open station indicator")
    result = get_station_recently_closed(result, nb_hours=4)
    logger.info("Create ratio bike filling on geo cluster station on time")
    result= filling_bike_on_geo_cluster(result, features_name='bikes_shift_'+str(freq)+'min')
#    logger.info("Create  Approximation (PAA) transformation") # Data Leak
#    result = get_paa_transformation(result, features_to_compute='probability', segments=10)
#    logger.info("Create  Approximation (SAX) transformation") # Data Leak
#    result = get_sax_transformation(result, features_to_compute='probability', segments=10, symbols=8)
    #logger.info("Create mean transformation") # Data Leak
#    result = create_rolling_mean_features(result, 
#                                          features_name='mean_6', 
#                                          feature_to_mean='probability', 
#                                          features_grp='station', 
#                                          nb_shift=6)
#    
    # logger.info("Create Bin hours of the day")
    # result['hours_binned'] = result.hour.apply(mapping_hours)
    
    #logger.info("Create interaction features with 'paa' and 'sax' ")
    #result = interaction_features('paa', 'sax', result)

    cut = date - pd.Timedelta(freq.replace('T', 'm'))
    stop = date + periods * pd.Timedelta(freq.replace('T', 'm'))
    logger.info("cut date %s", cut)
    logger.info("stop date %s", stop)
    
    train = result[result.index <= cut].copy()
    if how == 'train_test_split':
        logger.info("split train and test according to a prediction date")
        train_X = train.drop([observation, "future"], axis=1)
        train_Y = train['future'].copy()
        # time window
        mask = np.logical_and(result.index >= date, result.index <= stop)
        test = result[mask].copy()
        test_X = test.drop([observation, "future"], axis=1)
        test_Y = test['future'].copy()
        return train_X, train_Y, test_X, test_Y
    elif how is None:
        logger.info("Split X and y DataFrame")
        X = train.drop([observation, "future"], axis=1)
        y = train['future'].copy()
        return X, y


def interaction_features(a, b, df):
    """
    Create interaction between 2 features (a and b)
    Return :
     - Minus (a-b)
     - multiply (a*b)
     - ratio (a/b)    
    """
    
    ## Minus
    minus_label = a+'_minus_'+b
    df[minus_label] = df[a] - df[b]
    
    ## Multiply
    milty_label = a+'_multi_'+b
    df[milty_label] = df[a] * df[b]
    
    ## Ratio 
    ratio_label = a+'_ratio_'+b
    df[ratio_label] = df[a] / df[b]
    
    return df

def mapping_hours(hours):
    """
    Mapping hours of day in 5 grp.
    """
    # if hours >= 0 and hours < 6:
    #     return 0 #("nuit")
    # elif hours >= 6 and hours < 10:
    #     return 1 #("matin boulot")
    # elif hours >= 10 and hours < 12:
    #     return 2 #("matin")
    # elif hours >= 12 and hours < 14:
    #     return 3 #("midi")
    # elif hours >= 14 and hours < 17:
    #     return 4 #("aprem")
    # elif hours >= 17 and hours < 21:
    #     return 5 #("retour boulot")
    # elif hours >= 21 and hours < 24:
    #     return 6 #("soire")
    if hours >= 0 and hours < 6:
        return 0 #("nuit")
    elif hours >= 6 and hours < 12:
        return 1 #("matin")
    elif hours >= 12 and hours < 14:
        return 2 #("midi")
    elif hours >= 14 and hours < 17:
        return 3 #("aprem")
    elif hours >= 18 and hours < 24:
        return 4 #("soire")

def get_statio_cluster_geo_ratio_open_by_time(df):
    """
    Create a ratio of open station on time and cluster station geo
    """
    # Count station geo cluster
    grp_cluster_station_geo = pd.DataFrame(df.groupby('station_cluster_geo')['station'].nunique()).reset_index()
    grp_cluster_station_geo.columns=['station_cluster_geo', 'nb_station_geo_cluster']

    grp_df = pd.DataFrame(df.groupby(['ts', 'station_cluster_geo'], as_index=False)['is_open'].sum())
    grp_df.columns = ['ts', 'station_cluster_geo', 'total_station_open']

    # merging 2 DataFrame
    grp_df = grp_df.merge(grp_cluster_station_geo, on='station_cluster_geo', how='left')

    grp_df['ratio_station_geo_cluster_open'] = grp_df['total_station_open'] / grp_df['nb_station_geo_cluster']

    df = df.merge(grp_df[['ts','station_cluster_geo', 'ratio_station_geo_cluster_open']], 
                    on=['ts', 'station_cluster_geo'], how='left')
    return df

def get_statio_ratio_open_by_time(df):
    """
    Create a ratio of open station on time
    """


    nb_station = df.station.nunique()

    grp_df = pd.DataFrame(df.groupby('ts', as_index=False)['is_open'].sum())
    grp_df.columns = ['ts', 'total_station_open']
    grp_df['ratio_station_open'] = grp_df['total_station_open'] / nb_station

    df = df.merge(grp_df[['ts', 'ratio_station_open']], on='ts', how='left')
    return df

def get_weather(df, how='learning', freq=None):
    """
    Match timeseries with weather data.
    df : [Dataframe]
    If type == learning :
        Matching with historitical data weather
    if type == forcast :
        Matching with forcast data. Freq must be fill with this opton
    freq : Timedelta ex : "1H"
    """

    df = df.reset_index()

    # Check params
    if how not in ['learning', 'forecast']:
        logger.error('Bad option for get_weather. You must choose between learning or forecast')
        return df

    if how == 'forecast' and freq is None:
        logger.error("For forecast option, we must specify freq. Ex freq='1H'")


    # Process for learning matching
    if how == 'learning':
        lyon_meteo = pd.read_csv('data/lyon_weather.csv', parse_dates=['date'])
        lyon_meteo.rename(columns={'date':'ts'}, inplace=True)

        # have to labelencode weather_desc
        LE = LabelEncoder()
        lyon_meteo['weather_desc'] = LE.fit_transform(lyon_meteo['weather_desc'])

        # Dump LabelEncoder
        joblib.dump(LE, 'model/Label_Encoder_Weather.pkl')

        # Resemple data on 10
        clean_lyon_meteo = lyon_meteo.resample("10T", on="ts").mean().bfill().reset_index()
        df = df.merge(clean_lyon_meteo[['ts', 'temp', 'humidity', 'weather_desc']], on='ts', how='left')
        
        df = df.sort_index()
        df = df.set_index('ts')
        return df

    # Process for forecast matching
    if how == 'forecast':
        lyon_forecast = pd.read_csv('data/lyon_forecast.csv', parse_dates=['forecast_at', 'ts'])
        lyon_forecast['delta'] = lyon_forecast['ts'] - lyon_forecast['forecast_at']

        # Filter on delta with freq
        lyon_forecast = lyon_forecast[lyon_forecast['delta'] == freq]
        lyon_forecast.drop_duplicates(subset=['ts', 'delta'], keep='first', inplace=True)
        
        # Label encode weather_desc
        LE = joblib.load('model/Label_Encoder_Weather.pkl') 
        lyon_forecast['weather_desc'] = LE.transform(lyon_forecast['weather_desc'])

        #Merging
        # We take the last forecast (on freq) using backward merging
        df = df.sort_values('ts')
        df_index_save = df.index # Savind index merge will destroy it
        df = pd.merge_asof(left=df, right=lyon_forecast[['ts','temp', 'humidity', 'weather_desc']], on='ts', direction='backward')
        df.index = df_index_save

        # Resorting as originaly (to don't loose y_test order)
        df = df.sort_index()
        df = df.set_index('ts')

        return df




def get_summer_holiday(df):
    """
    Create bool for summer holiday (2017-09-04)
    """

    df['date'] = df.ts.dt.date
    df['date'] = df['date'].astype('str')

    # Create DF with unique date (yyyy-mm-dd)
    date_df = pd.DataFrame(df.date.unique(), columns=['date'])
    date_df['date'] = date_df['date'].astype('str')

    date_df['is_holiday'] = date_df['date'].apply(lambda x : parser.parse(x) < parser.parse("2017-09-04"))
    date_df['is_holiday'] = date_df['is_holiday'].astype('int')

    #merging
    df = df.merge(date_df, on='date', how='left')
    df.drop('date', axis=1, inplace=True)
    return df


def get_public_holiday(df, count_day=None):
    """
    Calcul delta with the closest holiday (count_day before and after) on absolute
    """
    df['date'] = df.ts.dt.date
    df['date'] = df['date'].astype('str')

    # Create DF with unique date (yyyy-mm-dd)
    date_df = pd.DataFrame(df.date.unique(), columns=['date'])
    date_df['date'] = date_df['date'].astype('str')
    # Create bool
    date_df['public_holiday'] = date_df.date.apply(lambda x: cal.is_holiday(parser.parse(x)))
    date_df['public_holiday'] = date_df['public_holiday'].astype(int)
    
    # Calcul the delta between the last public_holiday == 1 (max count_day)
    if count_day is not None:
        logger.info("compute delta with  public holiday on '%s' days", count_day)
        dt_list = []
        for holyday_day in date_df[date_df.public_holiday == 1].date.unique():
            for i in range(-count_day, count_day+1, 1):
                new_date = parser.parse(holyday_day) + timedelta(days=i)
                new_date_str = new_date.strftime("%Y-%m-%d")
                dt_list.append({'date' : new_date_str,
                                'public_holiday_count' : np.abs(i)})
        # DataFrame
        df_date_count = pd.DataFrame(dt_list)
        # Merging
        date_df = date_df.merge(df_date_count, on='date', how='left')
        # Filling missing value
        date_df['public_holiday_count'] = date_df['public_holiday_count'].fillna(0)

    #merging
    df = df.merge(date_df, on='date', how='left')
    df.drop('date', axis=1, inplace=True)
    return df

def get_sax_transformation(df, features_to_compute='probability', segments=10, symbols=8):
    """
    Re sort dataframe station / ts
    Aggr time serie for each station
    Symbolic Aggregate approXimation
    If the time serie can't be divide by segment. We take lhe last x value en df
    df : DataFrame
    features_to_compute : string - column's name of the features we want to agg
    segments : int - number of point we want to agg.
    symbols : int - Number of SAX symbols to use to describe slopes
    """

    sax_list_result = []
    df = df.reset_index()
    df = df.sort_values(['station', 'ts'])

    for station in df.station.unique():
        data = df[df.station == station].copy()
        n_paa_segments = round((len(data) * segments / 100) -0.5)
        n_sax_symbols_avg = round((len(data) * symbols / 100) -0.5)
        n_sax_symbols_slope = round((len(data) * symbols  / 100) -0.5)
        one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                        alphabet_size_slope=n_sax_symbols_slope)

        sax_list_result.extend(one_d_sax.inverse_transform(one_d_sax.fit_transform(data[features_to_compute][0:n_paa_segments * segments].values)).ravel())
        if len(sax_list_result) != len(data):
            sax_list_result.extend(data[features_to_compute][n_paa_segments * segments : len(data)].values)

        result = sax_list_result
            
    df['sax'] = result
    df['sax'] = df['sax'].astype('float')
    df = df.sort_values(['ts', 'station']) 
    df = df.set_index('ts')
    return df

def get_paa_transformation(df, features_to_compute='probability', segments=10):
    """
    Re sort dataframe station / ts
    Aggr time serie for each station
    Take the mean of each segment
    If the time serie can't be divide by segment. We add the last mean agg.
    df : DataFrame
    features_to_compute : string - column's name of the features we want to agg
    semgnets : int - number of point we want to agg.
    """
    paa_list_result = []
    df = df.reset_index()
    df = df.sort_values(['station', 'ts'])

    for station in df.station.unique():
        data = df[df.station == station]
        n_paa_segments = round((len(data) * segments / 100) -0.5)
        paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
        paa_inv_transf = np.repeat(paa.fit_transform(data[features_to_compute].values)[0], segments, axis=0)
        
        if len(data) != len(paa_inv_transf):
            nb_to_add = len(data) - len(paa_inv_transf)
            value_to_add = np.repeat(np.mean(data[features_to_compute].values[-nb_to_add:]), nb_to_add, axis=0) # Take the last X one and mean it
            result = np.append(paa_inv_transf, value_to_add) # Append regular paa and last segment mean
            paa_list_result.extend(result)
            
        else:
            result = paa_inv_transf
            paa_list_result.extend(result)
            
    df['paa'] = paa_list_result
    df['paa'] = df['paa'].astype('float')
    df = df.sort_values(['ts', 'station']) 
    df = df.set_index('ts')
    return df

def create_rolling_mean_features(df, features_name, feature_to_mean, features_grp, nb_shift):
    """
    function to create a rolling mean on "feature_to_mean" called "features_name" 
    groupby "features_grp" on "nb_shift" value
    Have to sort dataframe and re sort at the end
    """
    
    df['ts'] = df.index
    df = df.sort_values(['station', 'ts'])


    # Create shift features
    df[features_name] = df.groupby(features_grp)[feature_to_mean].apply(lambda x: x.rolling(window=nb_shift, min_periods=1).mean())

    df = df.sort_values(['ts', 'station']) 
    df = df.set_index('ts')
    return df

def cluster_station_geo_lyon(df):
    """
    Get Lyon station's cluster (from notebook)
    """
    cluster_lyon_geo = pd.read_csv('data/station_cluster_geo_armand.csv')
    df = df.merge(cluster_lyon_geo, on='station', how='inner')
    return df

def cluster_station_lyon(df):
    """
    Get Lyon station's cluster (from notebook)
    """
    cluster_lyon = pd.read_csv('data/cluster_lyon_armand.csv')
    df = df.merge(cluster_lyon, on='station', how='inner')
    return df

def create_shift_features(df, features_name, feature_to_shift, features_grp, nb_shift):
    """
    function to create shift features
    Have to sort dataframe and re sort at the end

    """
    df['ts'] = df.index
    df = df.sort_values(['station', 'ts'])


    # Create shift features
    df[features_name] = df.groupby([features_grp])[feature_to_shift].shift(nb_shift)
    df[features_name] = df[features_name].fillna(method='bfill')

    df.drop('ts', axis=1, inplace=True)
    return df

def create_cumul_trend_features(df, features_name):
    """
    Create cumulative features on trending bike station
    """
    df['ts'] = df.index
    df = df.sort_values(['station', 'ts'])

    df['bool_trend_sup'] = 0
    df.loc[df['bikes'] > df[features_name], 'bool_trend_sup'] = 1
    df['bool_trend_inf'] = 0
    df.loc[df['bikes'] < df[features_name], 'bool_trend_inf'] = 1
    df['bool_trend_equal'] = 0
    df.loc[df['bikes'] == df[features_name], 'bool_trend_equal'] = 1

    df = df.sort_values(['station', 'ts'])

    df['cumsum_trend_sup'] = df["bool_trend_sup"].groupby((df["bool_trend_sup"] == 0).cumsum()).cumcount()
    df['cumsum_trend_inf'] = df["bool_trend_inf"].groupby((df["bool_trend_inf"] == 0).cumsum()).cumcount()
    df['cumsum_trend_equal'] = df["bool_trend_equal"].groupby((df["bool_trend_equal"] == 0).cumsum()).cumcount()

    df.drop(['bool_trend_sup', 'bool_trend_inf', 'bool_trend_equal'], axis=1, inplace=True)

    df = df.sort_values(['ts', 'station']) 
    df = df.set_index('ts')
    return df

def get_station_recently_closed(df, nb_hours=4):
    """
    Create a indicator who check the number of periods the station was close during the nb_hours
    - 0 The station was NOT closed during nb_hours
    - > 1 The station was closes X times during nb_hours

    Need to sort the dataframe
    Warning : depend of the périod of resampling
    """
    # Resorting
    df = df.reset_index()
    df = df.sort_values(['station', 'ts'])
    time_period = nb_hours * 6 # For a 10T resempling, 1 hours -> 6 rows
    df['was_recently_open'] = df['is_open'].rolling(window=time_period, min_periods=1).sum()

    df = df.sort_values(['ts', 'station']) 
    df = df.set_index('ts')

    return df

def filling_bike_on_geo_cluster(df, features_name):
    """
    Get filling bike station on station Geo
    Calcul number of total stand on station Geo / time
    Calcul number of bike on station geo / time
    Create ratio on total stand and bike on station (shift) on geo station
    Merge the result with the DataFrame
    """

    df['ts'] = df.index
    # Total stand for station
    df['total_stand'] = df['bikes'] + df['stands']

    # Total stand by time and geo cluster
    total_stand_by_geo_cluster = df.groupby(['ts', 'station_cluster_geo'], as_index=False)['total_stand'].sum()
    total_stand_by_geo_cluster.rename(columns={'total_stand':'total_stand_geo_cluster'}, inplace=True)

    # Total bike by time and gro cluster (taking the shift features bike)
    features_shift_by_geo_cluster = df.groupby(['ts', 'station_cluster_geo'], as_index=False)[features_name].sum()
    features_shift_by_geo_cluster.rename(columns={features_name:features_name+'_geo_cluster'}, inplace=True)

    # Merging this 2 DataFrame
    grp_features_geo_cluster = total_stand_by_geo_cluster.merge(features_shift_by_geo_cluster, 
                                                                on=['ts', 'station_cluster_geo'], 
                                                                how='inner')

    # Create Ratio
    grp_features_geo_cluster['filling_station_by_geo_cluster'] = grp_features_geo_cluster[features_name+'_geo_cluster'] / grp_features_geo_cluster['total_stand_geo_cluster']
    grp_features_geo_cluster = grp_features_geo_cluster[['ts', 'station_cluster_geo', 'filling_station_by_geo_cluster']]
    # Merge with df
    df = df.merge(grp_features_geo_cluster, on=['ts', 'station_cluster_geo'], how='inner')
    #df = df.drop('total_stand', axis=1)
    df = df.sort_values(['ts', 'station']) 
    df = df.set_index('ts')
    return df

def fit(train_X, train_Y, test_X, test_Y, param, num_round=25):
    """Train the xgboost model

    Return the booster trained model
    """
    logger.info("fit")
    # param = {'objective': 'reg:linear'}

    #if param
    #param = {'objective': 'reg:logistic'}
    #param['eta'] = 0.2
    #param['max_depth'] = 4 # 6 original
    #param['silent'] = 1
    #param['nthread'] = 4
    # used num_class only for classification (e.g. a level of availability)
    # param = {'objective': 'multi:softmax'}
    # param['num_class'] = train_Y.nunique()
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, num_round, watchlist)
    return bst


def prediction(bst, test_X, test_Y):
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    pred = bst.predict(xg_test)
    return pred


def error_rate(bst, test_X, test_Y):
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
    logger.info('Test error using softmax = %s', error_rate)
    return error_rate


if __name__ == '__main__':
    DATAFILE = "./data/lyon.csv"
    THRESHOLD = 3

    raw = datareader(DATAFILE)
    df_clean = cleanup(raw)
    bonus = extract_bonus_by_station(df_clean)
    df_clean = df_clean.drop("bonus", axis=1)
    # df = (df_clean.pipe(time_resampling)
    #       .pipe(complete_data)
    #       .pipe(lambda x: availability(x, THRESHOLD)))
    df = (df_clean.pipe(time_resampling)
          .pipe(complete_data)
          .pipe(bikes_probability))

    # Note: date range is are 2017-07-08 15:20:28  -  2017-09-26 14:58:45
    start = pd.Timestamp("2017-07-11") # Tuesday
    # predict_date = pd.Timestamp("2017-07-26T19:30:00") # wednesday
    predict_date = pd.Timestamp("2017-07-26T10:00:00") # wednesday
    # predict the further 30 minutes
    freq = '30T'
    train_X, train_Y, test_X, test_Y = prepare_data_for_training(df,
                                                                 predict_date,
                                                                 freq=freq,
                                                                 start=start,
                                                                 periods=2,
                                                                 observation='probability')
    # train_X, train_Y, test_X, test_Y = prepare_data_for_training(df, predict_date, freq='1H', start=start, periods=2)

    bst = fit(train_X, train_Y, test_X, test_Y)
    # err = error_rate(bst, test_X, test_Y)
    # print("Error rate: {}".format(err))
    pred = prediction(bst, test_X, test_Y)
    rmse = np.sqrt(np.mean((pred - test_Y)**2))
    print("RMSE: {}".format(rmse))

    # put observation and prediction in a 'test' DataFrame
    test = test_X.copy()
    #obs = test_Y.to_frame()
    test['ts_future'] = test_Y.index.shift(1, freq=freq)

    test['observation'] = test_Y.copy()
    test['ts_future'] = test_Y.index.shift(1, freq=freq)
    test['prediction'] = pred
    test['error'] = pred - test_Y
    test['relative_error'] = 100. * np.abs(pred - test_Y) / test_Y
    test['quad_error'] = (pred - test_Y)**2
    test.to_csv("prediction-freq-{}-{}.csv".format(freq, predict_date))
