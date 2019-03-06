import pandas as pd
from tqdm import tqdm
tqdm.pandas()

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./input/test.csv")
print("Train shape : ", train.shape)
print("Test shape : ", train.shape)