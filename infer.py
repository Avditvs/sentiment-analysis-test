import os

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from tqdm.auto import tqdm

from utils.preprocessing import make_labels, tokenize
from utils.classes import SentimentDataset

DATA_DIR = "./data"
TEST_PATH = os.path.join(DATA_DIR, "test_cleaned.csv")

MODEL = "xlm-roberta-base"
MODEL_PRETRAINED = "./models/xlm_roberta_classif/checkpoint-1758"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PRETRAINED, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

data_test = pd.read_csv(TEST_PATH, na_filter=False)

x_test = tokenizer(list(data_test.content), max_length=128, padding=True, truncation=True)



training_args = TrainingArguments(
    "bert_base_uncased_classif",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=32,
    fp16 = True,
    fp16_opt_level = 'O1',
    evaluation_strategy = 'epoch',
    save_strategy="epoch",
    num_train_epochs=4
    
)

trainer = Trainer(
    model=model,
    args=training_args,
)

dataset_torch = SentimentDataset(x_test, [1 for _ in range(len(data_test))])

results = trainer.predict(dataset_torch)
predictions = np.argmax(results.predictions, axis=1)

data_test = data_test.assign(prediction=pd.Series(predictions).values)
data_test.to_csv(os.path.join(DATA_DIR, "predictions.csv"), index = False)
