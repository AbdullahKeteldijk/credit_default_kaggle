import pandas as pd
from src.model_pipeline import ModelPipeline

def main():

    df = pd.read_csv('../hmeq.csv')

    pipe = ModelPipeline(df)
    pipe.fit_transform()
    pipe.print_model_performance()


if __name__ == '__main__':
    main()
