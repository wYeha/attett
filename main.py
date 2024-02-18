import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import random
lst = ['robot'] *  10
lst += ['human'] *  10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI': lst})

ohe = OneHotEncoder()

one_hot_encoded = ohe.fit_transform(data[['whoAmI']]).toarray()

one_hot_df = pd.DataFrame(one_hot_encoded, columns=ohe.get_feature_names_out(['whoAmI']))

print(one_hot_df.head())