from sklearn.naive_bayes import GaussianNB

class NaiveBayesClassifier:
    def train(self, X_train, y_train, feature_mask):
        nb_model = GaussianNB()
        X_train_selected = X_train.values[:, feature_mask==1]
        nb_model.fit(X_train_selected, y_train)
        return nb_model

    def test(self, nb_model, X_test, feature_mask):
        X_test_selected = X_test.values[:, feature_mask==1]
        y_pred = nb_model.predict(X_test_selected)
        return y_pred

