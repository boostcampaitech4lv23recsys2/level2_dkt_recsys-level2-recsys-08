import pandas as pd

from catboost_feature import Features

class CBDataset:
    def __init__(self, data_path):
        features = Features()
        self.features = features.FEAT
        self.cat_features = features.CAT_FEAT
        self.num_features = list(set(self.features) - set(self.cat_features))
        self.df = pd.read_pickle(data_path)
        
        self.descript = f"{len(self.features)}_features"
    
    def split_data(self, pseudo_labeling=False):
        self._convert_cat_features_dtype()
        self._convert_num_features_dtype()
        
        train_valid_df = self.get_train_data(pseudo_labeling)
        
        valid = train_valid_df[(train_valid_df.userID != train_valid_df.userID.shift(-1)) &
                               (train_valid_df["kind"] == "train")]
        train = train_valid_df[~train_valid_df.index.isin(valid.index)]
        
        X_train, y_train = train[self.features], train["answerCode"]
        X_valid, y_valid = valid[self.features], valid["answerCode"]
        
        return X_train, X_valid, y_train, y_valid
    
    def get_train_data(self, pseudo_labeling=False):
        if pseudo_labeling:
            # self.df[self.df.answer == -1] = correct answer
            pass
        return self.df[self.df.answerCode != -1]
    
    def get_test_data(self):
        return self.df[self.df.answerCode == -1][self.features]
    
    def _convert_cat_features_dtype(self):
        for cat_feat in self.cat_features:
            self.df[cat_feat] = self.df[cat_feat].astype("category")
    
    def _convert_num_features_dtype(self):
        for num_feat in self.num_features:
            self.df[num_feat] = self.df[num_feat].astype(float)