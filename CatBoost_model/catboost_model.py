from catboost import CatBoostClassifier

class CatBoostModel:
    def __init__(self, args, output_dir):
        self.output_dir = output_dir
        self.model = CatBoostClassifier(iterations=args.iteration,
                                        random_seed=args.seed,
                                        custom_metric=["AUC", "Accuracy"],
                                        eval_metric="AUC",
                                        early_stopping_rounds=args.early_stopping,
                                        train_dir=self.output_dir,
                                        learning_rate=args.lr,
                                        
                                        task_type="GPU",
                                        devices="0")
    
    def fit(self, X, y, cat_features, eval_set, verbose=100):
        self.model.fit(X, y, 
                       cat_features=cat_features, 
                       eval_set=eval_set,
                       verbose=verbose)
    
    def inference(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def gat_feature_importances(self):
        return self.model.feature_importances_

    def gat_feature_names(self):
        return self.model.feature_names_

    def gat_best_score(self):
        return self.model.best_score_

    def gat_best_iter(self):
        return self.model.best_iteration_

    def save_model_(self):
        return self.model.save_model('model')
