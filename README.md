## Essentials_UZH
Public Repository for the 2024 edition of Essentials in Text and Speech Processing (06SM521-900) at the University of Zurich
[(Course Information)](https://studentservices.uzh.ch/uzh/anonym/vvz/index.html#/details/2024/003/SM/51079434/50000003/Wirtschaftswissenschaftliche%2520Fakult%25C3%25A4t/51085510/Master%2520of%2520Science%2520UZH%2520in%2520Informatik%2520(RVO22)/51087487/Data%2520Science)

### Project Overview

This project focuses on classifying cryptocurrency pump-and-dump messages from Telegram into specific categories, notably:

- **Pump Announcements**
- **Countdown Messages**
- **Coin Release Announcements**
- **Pump Results**
- **Delay or Cancellation Notices**
- **Unrelated**

We apply various models and techniques to achieve accurate message classification, from heuristic-based methods, zero-shot classifications to finetuning different language models. In addition, we implement a **snowball search** mechanism to automatically identify new potential pump channels based on observed connections and activity. 

To facilitate timely alerts, we also develop a **real-time pipeline** where exchanges can subscribe to receive warning notifications as they happen. For more details on subscription, see [this link](https://t.me/EssentialsWarningChannel).

### Setup

1. Create environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create .env and provide the necessary credentials. If there is no .env file, one will be automatically created from a template.

### Directory Structure

#### Folders

- **channel_discovery/**
  - Contains the code for the SnowballSearch.

- **data/**
  - **./external**: Data from Hu et al. (2023), manually labeled by us.
  - **./internal**: Data either scraped of created by us.

- **preprocessing/**
  - Contains the code for the Telegram message preprocessing.

- **utils/**
  - Contains various helper functions such as the API key management or the proxy set up for scraping.

### Files

#### 01a_create_exchange_list.py
Python script to fetch a list of crypto exchange names. The list is later used in the preprocessing of the Telegram messages to tag exchange names with special tokens (_CEX or _DEX).

#### 01b_message_scraping.py
Python script to scrape the complete message history for a given list of P&D channels. The scraped messages are stored under 'data/internal/raw'. If run multiple times, it only updates the existing files with new messages instead of scraping the complete history again.

```bash
python 01b_message_scraping.py
```

#### 01c_message_preprocessing.py
Python script applying the preprocessing to the scraped messages. Contains various methods of text normalization to harmonize messages across different channels.

```bash
python 01c_message_preprocessing.py
```

- **StandardNormalizer Class**  
  A class containing all text processing steps, already initialized with a list of exchanges and regex patterns. Uses the TweetTokenizer from nltk.

#### 01d_combine_training_data.ipynb
Jupyther notebook which was used to combine all manually labeled and synthetically created data sources into one training set later used to train and evaluate various language models.

#### 02a_heuristic_baseline_model.ipynb
This Jupyter notebook implements a heuristic-based baseline model to classify the messages.


#### 02b_word_embeddings.ipynb
This Jupyter notebook implements an embedding-based model to classify the messages. The embeddings are based on GloVe and are processed using a custom built neural network.


#### 03a_k_fold_evaluation.py
This Python script containes the code to perform the fine-tuning of the four models tested (DistilBERT, BERT-base, BERTweet, and RoBERTa) in a k-fold cross validation to robustly compare the performance. Requires access to a GPU.

Depending on the GPU, the script will run for several hours. To ensure that it runs continuously it should be run with nohup:

```bash
nohup python 03a_k_fold_evaluation.py &
```

#### 03b_train_bertweet.py
This Python script was used to fine-tune the best-performing model found in 03a (BERTweet) and perform hyperparameter tuning.

```bash
nohup python 03b_train_bertweet.py &
```

#### 04a_gpt4o_zeroshot.ipynb
This Jupyter notebook classifies the messages using the OpenAI API. The classification is conducted zero-shot.


#### 04b_gpt4o_zeroshot_with_class_definition.ipynb
This Jupyter notebook classifies the messages using the OpenAI API. The classification is conducted by providing an exact definition of the distinct classes.


#### 04c_gpt4o_fewshot.ipynb
This Jupyter notebook classifies the messages using the OpenAI API. The classification is conducted by providing a few real-world examples for each distinct class.


#### 05a_snowball_search.py
This script recursively explores links in Telegram messages to discover new pump channels. It classifies channels as pump-related if pump messages exceed a certain configurable threshold. Newly discovered channels are added to a list to expand the search network and avoid duplicate searches.


#### 06a_real_time_pipeline.py

This script sets up a real-time pipeline to monitor Telegram channels for pump-and-dump messages and send notifications to exchanges. When run, the script currently automatically subscribes the given number in the .env file to all monitored Telegram channels since this is a requirement for the event listener to work. Use with care!

```bash
nohup python 05a_real_time_pipeline.py &
```

- **ExchangeWarner Class**  
  A class to monitor specified Telegram channels for messages related to cryptocurrency pumps, using a classifier model to identify key message types (e.g., pump announcements, coin releases, cancellations).

- **Key Features**  
  - **Telegram Client**: Connects to Telegram using a client and monitors channels for specific message patterns.
  - **Message Classification**: Uses a text classifier to categorize messages, identifying pump announcements, cancellations, and coin releases.
  - **Notifications**: Sends formatted alerts to subscribed channels based on detected pump signals.
  - **Channel Subscription**: Can automatically join and subscribe to new pump channels as they are identified.

- **Pipeline Workflow**  
  1. **Monitor Channels**: Watches specified channels for incoming messages.
  2. **Classify Messages**: Uses the `pump_bertweet` model to classify messages based on predefined categories.
  3. **Extract Exchange Names**: Identifies exchange names in messages and manages a cache for tracking exchanges associated with each channel.
  4. **Send Notifications**: Sends structured notifications to the designated target channel with relevant details like confidence scores and exchange names.

This pipeline provides real-time insights into pump activity, enabling exchanges to react promptly to pump-and-dump alerts.

