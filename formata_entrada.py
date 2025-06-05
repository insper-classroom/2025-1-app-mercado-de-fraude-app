import pandas as pd
import numpy as np
import os

def create_refined(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DataFrame de transações (transactions) e retorna um DataFrame 'refined'
    contendo as features temporais e de deslocamento calculadas.
    """
    # 1) Carrega os dados de pagadores e vendedores
    df_payers = pd.read_feather('data/payers_raw.feather')
    df_sellers = pd.read_feather('data/seller_raw.feather')

    # 2) Merge com pagadores
    df = df_transactions.merge(
        df_payers,
        left_on='card_id',
        right_on='card_hash',
        how='left',
        suffixes=('', '_payer')
    )

    # 3) Merge com vendedores
    df = df.merge(
        df_sellers,
        on='terminal_id',
        how='left',
        suffixes=('', '_seller')
    )

    # 4) Seleção das colunas necessárias para o raw_merge
    desired = [
        'transaction_id',
        'tx_datetime',
        'tx_date',
        'tx_time',
        'tx_amount',
        'card_id',
        'card_bin',
        'card_first_transaction',
        'terminal_id',
        'latitude',
        'longitude',
        'terminal_operation_start',
        'terminal_soft_descriptor'
    ]
    df_raw_merge = df[desired].copy()

    # 5) Conversão de latitude/longitude para float
    df_raw_merge['latitude']  = df_raw_merge['latitude'].astype(float)
    df_raw_merge['longitude'] = df_raw_merge['longitude'].astype(float)

    # 6) Conversão de datas e extração de features temporais
    df_raw_merge['tx_datetime']              = pd.to_datetime(df_raw_merge['tx_datetime'])
    df_raw_merge['card_first_transaction']   = pd.to_datetime(df_raw_merge['card_first_transaction'])
    df_raw_merge['terminal_operation_start'] = pd.to_datetime(df_raw_merge['terminal_operation_start'])

    df_raw_merge['tx_hour'] = df_raw_merge['tx_datetime'].dt.hour
    df_raw_merge['tx_dow']  = df_raw_merge['tx_datetime'].dt.dayofweek

    df_raw_merge['card_age_days'] = (
        df_raw_merge['tx_datetime'] - df_raw_merge['card_first_transaction']
    ).dt.days
    df_raw_merge['terminal_age_days'] = (
        df_raw_merge['tx_datetime'] - df_raw_merge['terminal_operation_start']
    ).dt.days

    # 7) Histórico do cartão: valor e intervalo entre transações
    df_raw_merge = df_raw_merge.sort_values(['card_id', 'tx_datetime'])
    df_raw_merge['tx_amount_prev'] = (
        df_raw_merge.groupby('card_id')['tx_amount']
        .shift(1)
        .fillna(0)
    )
    df_raw_merge['hours_since_prev'] = (
        df_raw_merge['tx_datetime'] - df_raw_merge.groupby('card_id')['tx_datetime'].shift(1)
    ).dt.total_seconds().div(3600).fillna(0)

    # 8) Cálculo de travel_speed via fórmula de Haversine
    df_raw_merge['prev_lat'] = df_raw_merge.groupby('card_id')['latitude'].shift(1)
    df_raw_merge['prev_lon'] = df_raw_merge.groupby('card_id')['longitude'].shift(1)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Raio da Terra em km
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    df_raw_merge['distance_km'] = haversine(
        df_raw_merge['prev_lat'],
        df_raw_merge['prev_lon'],
        df_raw_merge['latitude'],
        df_raw_merge['longitude']
    )
    df_raw_merge['travel_time_h'] = df_raw_merge['hours_since_prev'].replace(0, np.nan)
    df_raw_merge['speed_kmh']     = df_raw_merge['distance_km'] / df_raw_merge['travel_time_h']
    df_raw_merge['travel_speed']  = df_raw_merge['speed_kmh']

    # 9) Seleção das colunas finais para o “refined”
    features = [
        'tx_datetime',
        'tx_amount',
        'tx_amount_prev',
        'hours_since_prev',
        'tx_hour',
        'tx_dow',
        'card_age_days',
        'terminal_age_days',
        'card_bin',
        'latitude',
        'longitude',
        'travel_speed'
    ]
    df_refined = df_raw_merge[features].reset_index(drop=True)

    return df_refined
