# Sentiment Analysis Test

- [Sentiment Analysis Test](#sentiment-analysis-test)
  - [🎯 Goals](#user-content--goals)
  - [📊 Datasets](#user-content--datasets)
  - [📖 Rules](#user-content--rules)
  - [👩‍💻 What to do](#user-content--what-to-do)

## 🎯 Goals

We want a multilingual text classifier that predicts the sentiment polarity of a given text.
Build two models that satisfy this task and choose the best one.
Please explain in detail each choice you made when building your models and how you choose the best one.

Possible sentiments:
* `positive`
* `negative`
* `neutral`

## 📊 Datasets

* `data/train.csv`: a training dataset containing 25k multilingual texts annotated with their corresponding sentiment
* `data/test.csv`: a test dataset containing 2500 multilingual texts

-------

## Work I've done

My answer to this test is oriented in 4 main parts in which I'll descrive more precicesly later, each one is done in a separated notebook that you can find in teh `notebooks` directory:
- Description of the problem and data analysis
- The different availables solutions
- The solutions I've implemented

To clean the datasets and make the train/val split, run ```python preprocess.py```
To run inference, run `python infer.py`

The predictions are located in `data/predictions.csv`

--------


## 📖 Rules

* Code should be written in Python 3
* Code should be easily runnable, provide a pip requirements.txt file or a conda environment.yml file describing code dependencies
* Code should be documented to explain your methodology, choices, and how to run and get results
* Code should output a file `predictions.csv`, containing the predictions of your best classifier on the test dataset

## 👩‍💻 What to do

1. Fork the project via `github`
2. Clone your forked repository project https://github.com/YOUR_USERNAME/sentiment-analysis-test
3. Commit and push your different modifications
4. Send us a link to your fork once you're done!
