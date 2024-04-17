# Toxicity Detection Model

## Overview

This repository contains code for training a toxicity detection model using Natural Language Processing (NLP) techniques. The model is trained on a dataset of tweets labeled for toxicity.

## Dataset

The dataset used for training the model (`FinalBalancedDataset.csv`) contains labeled tweets along with their toxicity scores. Before training the model, the data is preprocessed to clean and tokenize the text, lemmatize words, and remove stopwords.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, matplotlib, nltk, scikit-learn

## Setup

1. Clone the repository:
   
   ```bash
   git clone https://github.com/nandini601/toxicity-detection.git
   cd toxicity-detection
