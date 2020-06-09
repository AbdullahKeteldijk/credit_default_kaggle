import unittest
import pandas as pd
from src.model_pipeline import ModelPipeline

class TestModelNames(unittest.TestCase):

    def test_modelled_prediction(self):
        df = pd.read_csv('../data/hmeq.csv')

        pipe = ModelPipeline(df)
        pipe.fit_transform()
        pipe.generate_model_name()

        self.assertEqual(pipe.model_name, "Ensemble model with modelled prediction")

    def test_mean_prediction(self):
        df = pd.read_csv('../data/hmeq.csv')

        pipe = ModelPipeline(df, mean_model=True)
        pipe.fit_transform()
        pipe.generate_model_name()

        self.assertEqual(pipe.model_name, "Ensemble model with mean prediction")

    def test_simple_model(self):
        df = pd.read_csv('../data/hmeq.csv')

        pipe = ModelPipeline(df, simple_model=True)
        pipe.fit_transform()
        pipe.generate_model_name()

        self.assertEqual(pipe.model_name, "Simple Logit model")


if __name__ == '__main__':
    unittest.main()
