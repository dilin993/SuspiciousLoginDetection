import pandas as pd
import feature_calculation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from joblib import dump, load

df = pd.read_csv('login_data.csv')
FEATURES_TO_SELECT = [feature_calculation.FEATURE_FAILURE_COUNT, feature_calculation.FEATURE_LOGIN_TIME,
                      feature_calculation.FEATURE_HOUR, feature_calculation.FEATURE_WEEKDAY,
                      feature_calculation.FEATURE_LONGITUDE, feature_calculation.FEATURE_LATITUDE]
X = df[FEATURES_TO_SELECT]
df.info()
# clustering = DBSCAN(eps=3, min_samples=2).fit(X)
clustering = KMeans(n_clusters=2, random_state=0).fit(X)
df['labels'] = clustering.labels_
print(df)
dump(clustering, 'model.joblib')