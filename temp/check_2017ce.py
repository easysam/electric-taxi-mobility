from utils import display
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == '__main__':
    display.configure_pandas()
    ce17 = pd.read_parquet('data/charging_event/charging_event_30min_200m_531station.parquet')
    ce17.info()
    print(ce17['license'].nunique(), 'vehicles')
    print(ce17)
    print(ce17['s_idx'].value_counts().head(30))
    print(ce17['s_idx'].nunique())
    print(ce17['s_idx'].value_counts().loc[ce17['s_idx'].value_counts() > 30].shape)
    ce17['s_idx'].value_counts().loc[ce17['s_idx'].value_counts() > 30].plot(kind='density')
    plt.show()
    print(ce17['license'].value_counts().loc[ce17['license'].value_counts() > 30].shape)
    ce17['license'].value_counts().loc[ce17['license'].value_counts() > 20].plot(kind='density')
    plt.show()
