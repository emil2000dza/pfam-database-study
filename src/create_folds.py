# create_folds.py
# import pandas and model_selection module of scikit-learn

import numpy as np
import pandas as pd
from sklearn import model_selection
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import config

if __name__ == "__main__":
    # Read training data
     df = pd.read_csv("../input/pfam_736.csv")
     df.family_id = df['family_accession'].map(config.MULTI_CLASS_DICT)

     # we create a new column called kfold and fill it with -1
     df["kfold"] = -1
     # the next step is to randomize the rows of the data
     df = df.sample(frac=1).reset_index(drop=True)

     # fetch labels
     y = df.family_id.values

     # initiate the kfold class from model_selection module
     kf = model_selection.StratifiedKFold(n_splits=5)

     # fill the new kfold column
     for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

     # save the new csv with kfold column
     df.to_csv("../input/pfam_736_folds.csv", index=False)