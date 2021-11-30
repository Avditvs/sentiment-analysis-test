import os

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import clean_dataset
from utils.preprocessing import detect_languages



DATA_DIR = "./data"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

data_train = pd.read_csv(TRAIN_PATH)
data_test = pd.read_csv(TEST_PATH)

data_train = clean_dataset(data_train)
data_test = clean_dataset(data_test)

data_train, _ = detect_languages(data_train)
data_test, _ = detect_languages(data_test)


data_train, data_val = train_test_split(data_train, test_size=.25)



paths = { k:os.path.join(DATA_DIR, f"{k}_cleaned.csv") for k in ["train", "val", "test"]}

data_train.to_csv(paths["train"])
data_val.to_csv(paths["val"])
data_test.to_csv(paths["test"])


