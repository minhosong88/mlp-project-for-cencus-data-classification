import unittest
from scripts.models import (
    TwoLayerPerceptron,
    ThreeLayerPerceptron,
    FourLayerPerceptron,
    FiveLayerPerceptron,
    FLPRMSProp,
    FLPAdaM
)
from scripts.load_and_preprocess import load_data, preprocess_data
from sklearn.metrics import accuracy_score


class TestMLPModels(unittest.TestCase):

    def setUp(self):
        df = load_data('data/acs2017_census_tract_data.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_data(
            df)

    def test_two_layer_perceptron(self):
        model = TwoLayerPerceptron(n_hidden=30, epochs=10)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertGreaterEqual(accuracy_score(self.y_test, y_pred), 0.5)

    def test_three_layer_perceptron(self):
        model = ThreeLayerPerceptron(n_hidden=30, epochs=10)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertGreaterEqual(accuracy_score(self.y_test, y_pred), 0.5)

    def test_four_layer_perceptron(self):
        model = FourLayerPerceptron(n_hidden=30, epochs=10)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertGreaterEqual(accuracy_score(self.y_test, y_pred), 0.5)

    def test_five_layer_perceptron(self):
        model = FiveLayerPerceptron(n_hidden=30, epochs=10)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertGreaterEqual(accuracy_score(self.y_test, y_pred), 0.5)

    def test_five_layer_perceptron_rmsprop(self):
        model = FLPRMSProp(n_hidden=30, epochs=10)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertGreaterEqual(accuracy_score(self.y_test, y_pred), 0.5)

    def test_five_layer_perceptron_adam(self):
        model = FLPAdaM(n_hidden=30, epochs=10)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertGreaterEqual(accuracy_score(self.y_test, y_pred), 0.5)


if __name__ == '__main__':
    unittest.main()
