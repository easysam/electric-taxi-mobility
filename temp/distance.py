import numpy as np
from sklearn.metrics.pairwise import haversine_distances

if __name__ == '__main__':
    a = haversine_distances(np.radians([[22.584299, 113.881500]]),
                            np.radians([[22.68484147237141, 114.12180202062079]]))
    print(a * 6371008.8)