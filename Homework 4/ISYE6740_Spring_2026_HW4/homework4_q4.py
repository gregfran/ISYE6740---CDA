import numpy as np
import pandas as pd
import os
# sklearn pkgs
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('data/marriage.csv')

    print(df.shape)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # results folder
    os.makedirs('results', exist_ok=True)
    load_data()
    pass

if __name__ == "__main__":
    main()