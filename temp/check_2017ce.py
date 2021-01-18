from utils import display
import pandas as pd
if __name__ == '__main__':
    display.configure_pandas()
    df = pd.read_parquet('data/charging_event/charging_event_30min_200m_531station.parquet')
    df.info()
    print(df['license'].nunique(), 'vehicles')
    print(df)
    print(df['s_idx'].value_counts().head(30))
    print(df['s_idx'].nunique())
