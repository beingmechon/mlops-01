import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data['total_usage'] = data['tenure'] * data['monthly_charges']
    return data
