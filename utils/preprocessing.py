import re

import pandas as pd
from tqdm.auto import tqdm
import langdetect



def remove_URL(text):
    reg = re.compile(r'https?://\S+|www\.\S+')
    return reg.sub(r'',text)

def remove_html(text):
    reg=re.compile(r'<.*?>')
    return reg.sub(r'',text)

def remove_mentions(text):
    reg = re.compile(r"@[A-Za-z0-9]+")
    return reg.sub(r'', text)

def remove_emoji(text):
    # https://en.wikipedia.org/wiki/Unicode_block
    reg = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])"
    )
    return reg.sub(r'', text)

def clean_text(text, limit = None):
    cleaning_funcs = [remove_html, remove_URL, remove_mentions, remove_emoji]
    
    for cleaning_func in cleaning_funcs:
        text = cleaning_func(text)
    if limit:
        text = text[:limit]
    return text

def clean_dataset(dataset, limit = None):
    cleaned_sentences = [clean_text(sentence, limit) for sentence in tqdm(dataset.content)]
    dataset = dataset.assign(content=pd.Series(cleaned_sentences).values)
    return dataset


def make_labels(input_labels, regress=False):
    labels = []
    for label in input_labels:
        if label == "positive":
            labels.append(1. if regress else 2)
        elif label=="neutral" or label=="unassigned":
            labels.append(.5 if regress else 1)
        else:
            labels.append(0. if regress else 0)
    return labels

def tokenize(tokenizer, sentences, length=128):
    return tokenizer(list(sentences), max_length=length, padding=True, truncation=True)

def detect_languages(data):
    languages_detected = []
    
    for sentence in tqdm(data.content):
        try:
            lang = langdetect.detect(sentence)
        except:
            lang = "undetected"
        languages_detected.append(lang)
        
    data = data.assign(language=pd.Series(languages_detected).values)
    return data, languages_detected