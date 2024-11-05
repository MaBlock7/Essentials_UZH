# Essentials_UZH
Public Repository for the UZH course Essentials in Text and Speech Processing (06SM521-900)

## Directory Structure

### Folders

- **channel_discovery/**
  - Purpose: Contains scripts related to identifying and managing channels for data scraping.
  - Notes: Renamed for organizational clarity.

- **data/**
  - Purpose: Stores real-time scraping data and test channels.
  - Notes: Added functionality for real-time data scraping.

- **preprocessing/**
  - Purpose: Contains scripts for preprocessing raw data for model readiness.
  - Notes: Added real-time scraping and test channel support.

- **utils/**
  - Purpose: Utility scripts used across various parts of the project.
  - Notes: Renamed for better grouping of related files.

### Files

- **.gitignore**
  - Purpose: Specifies files and directories to be ignored by Git.
  - Notes: Updated to reflect new file organization.

- **01a_create_exchange_list.py**
  - Purpose: Script for creating a list of exchanges for data collection.
  - Notes: Renamed for consistency.

- **01b_message_scraping.py**
  - Purpose: Script for scraping messages from identified channels or sources.
  - Notes: Renamed for consistency.

- **01c_message_preprocessing.py**
  - Purpose: Preprocesses scraped messages for use in model training.
  - Notes: Renamed for consistency.

- **01d_combine_training_data.ipynb**
  - Purpose: Jupyter notebook to combine and format training data from various sources.
  - Notes: Renamed for consistency.

- **02a_heuristic_baseline_model.ipynb**
  - Purpose: Jupyter notebook for building a heuristic-based baseline model.
  - Notes: Renamed for clarity and organization.

- **02b_word_embeddings.ipynb**
  - Purpose: Notebook for generating and using word embeddings.
  - Notes: Renamed for clarity and consistency.

- **03a_k_fold_evaluation.py**
  - Purpose: Script for performing k-fold evaluation on model performance.
  - Notes: Renamed for consistency.

- **03b_train_bertweet.py**
  - Purpose: Script for training a BERTweet model on prepared data.
  - Notes: Renamed for clarity.

- **04a_gpt4o_zeroshot.ipynb**
  - Purpose: Jupyter notebook for running zero-shot classification with GPT-4.
  - Notes: Renamed for clarity.

- **04b_gpt4o_zeroshot_with_class_definition.ipynb**
  - Purpose: Notebook for GPT-4 zero-shot classification with class definitions.
  - Notes: Renamed for consistency.

- **04c_gpt4o_fewshot.ipynb**
  - Purpose: Jupyter notebook for few-shot classification using GPT-4.
  - Notes: Renamed for clarity and organization.

- **05a_real_time_pipeline.py**
  - Purpose: Script for the real-time data processing pipeline.
  - Notes: Renamed for better organization.

- **README.md**
  - Purpose: Provides an overview of the project.
  - Notes: Initial commit.

- **anon.session**
  - Purpose: Contains session data for notification case-study.
  - Notes: Added notification case-study functionality.

- **requirements.txt**
  - Purpose: Lists Python dependencies required for the project.
  - Notes: Updated to include data labeling scripts.
