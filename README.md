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


### Directory Structure

#### Folders

- **channel_discovery/**
  - 

- **data/**
  - 

- **preprocessing/**
  - 

- **utils/**
  - 

### Files

#### 01a_create_exchange_list.py


#### 01b_message_scraping.py


#### 01c_message_preprocessing.py


#### 01d_combine_training_data.ipynb


#### 02a_heuristic_baseline_model.ipynb
This Jupyter notebook implements a heuristic-based baseline model to classify the messages.


#### 02b_word_embeddings.ipynb
This Jupyter notebook implements an embedding-based model to classify the messages. The embeddings are based on GloVe and are processed using a custom built neural network.


#### 03a_k_fold_evaluation.py

#### 03b_train_bertweet.py

#### 04a_gpt4o_zeroshot.ipynb
This Jupyter notebook classifies the messages using the OpenAI API. The classification is conducted zero-shot.


#### 04b_gpt4o_zeroshot_with_class_definition.ipynb
This Jupyter notebook classifies the messages using the OpenAI API. The classification is conducted by providing an exact definition of the distinct classes.


#### 04c_gpt4o_fewshot.ipynb
This Jupyter notebook classifies the messages using the OpenAI API. The classification is conducted by providing a few real-world examples for each distinct class.


#### 05a_real_time_pipeline.py

This script sets up a real-time pipeline to monitor Telegram channels for pump-and-dump messages and send notifications to exchanges.

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

