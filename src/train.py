import pandas as pd
from src.model_pipeline import ModelPipeline


def main():

    df = pd.read_csv('../data/hmeq.csv')
    pipe = ModelPipeline(df, simple_model=True)
    pipe.fit_transform()
    pipe.print_model_performance(print_output=False)
    pipe.save_output('../models/output.json')

if __name__ == '__main__':
    main()