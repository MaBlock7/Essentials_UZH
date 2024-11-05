import copy
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)


MODEL_DICT = {
    'distilbert_cased': {
        'model': AutoModelForSequenceClassification.from_pretrained(
            'distilbert/distilbert-base-cased',
            num_labels=6,
            problem_type="single_label_classification"
        ),
        'tokenizer': AutoTokenizer.from_pretrained('distilbert/distilbert-base-cased'),
        'max_len': 128
    },
    'bert_cased': {
        'model': AutoModelForSequenceClassification.from_pretrained(
            'google-bert/bert-base-cased',
            num_labels=6,
            problem_type="single_label_classification"
        ),
        'tokenizer': AutoTokenizer.from_pretrained('google-bert/bert-base-cased'),
        'max_len': 128
    },
    'bertweet': {
        'model': AutoModelForSequenceClassification.from_pretrained(
            'vinai/bertweet-base',
            num_labels=6,
            problem_type="single_label_classification"
        ),
        'tokenizer': AutoTokenizer.from_pretrained('vinai/bertweet-base'),
        'max_len': 128
    },
    'roberta_large': {
        'model': AutoModelForSequenceClassification.from_pretrained(
            'FacebookAI/roberta-large',
            num_labels=6,
            problem_type="single_label_classification"
        ),
        'tokenizer': AutoTokenizer.from_pretrained('FacebookAI/roberta-large'),
        'max_len': 128
    }
}


def prevent_data_slippage(train_df, test_df, col_name='message'):
    original_len = len(test_df)
    test_df = test_df[~(test_df.message.isin(train_df.message))]
    print(f'Removed {original_len - len(test_df)} entries from the test set!')
    return train_df, test_df


def tokenize_data(texts, tokenizer, max_len):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_len)


class MessagesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, roberta):
        self.encodings = tokenize_data(texts, tokenizer, max_len)
        self.labels = labels
        self.roberta = roberta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long if self.roberta else torch.int32)
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Ensure labels are of type torch.long
        labels = labels.type(torch.long)

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute weighted loss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    Compute various evaluation metrics for classification models.

    Args:
        eval_pred (tuple): A tuple containing the logits and the true labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    logits, labels = eval_pred

    # Get the predicted class indices from logits
    predictions = np.argmax(logits, axis=1)

    # Calculate metrics
    f1_average = f1_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average=None)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    return {
        'f1 weighted average': f1_average,
        'precision': precision,
        'recall': recall,
        'f1 class 0': f1[0],
        'f1 class 1': f1[1],
        'f1 class 2': f1[2],
        'f1 class 3': f1[3],
        'f1 class 4': f1[4],
        'f1 class 5': f1[5],
    }


def main():

    # Read in the training data
    df = pd.read_csv('./data/internal/training_data/training_data_no_duplicates_per_channel.csv')
    # Drop all entries that do not have a text for their economic activity
    df = df.dropna(subset='message')
    print(len(df))

    # Calculate the class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    class_weights_list = [class_weights_dict[i] for i in range(len(class_weights_dict))]
    print(class_weights_dict)

    # Initialize Stratified K-Fold with 3 splits
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store results for each model across folds
    cv_results = {}

    for model_name, config in MODEL_DICT.items():
        print("-" * 50)
        print(f"Starting 5-Fold Cross-Validation for {model_name}")

        # To store fold metrics
        fold_metrics = []

        # Loop over each fold
        for fold, (train_index, val_index) in enumerate(kf.split(df, df['label'])):
            print(f"Fold {fold + 1}")

            # Split the data into training and validation sets
            train_df, val_df = df.iloc[train_index], df.iloc[val_index]

            # Remove any data leakage between train and val
            train_df, val_df = prevent_data_slippage(train_df, val_df)

            # Create datasets
            train_dataset = MessagesDataset(
                train_df['message'].tolist(),
                list(train_df['label']),
                config['tokenizer'],
                config['max_len'],
                roberta=(model_name == 'roberta_large')
            )

            val_dataset = MessagesDataset(
                val_df['message'].tolist(),
                list(val_df['label']),
                config['tokenizer'],
                config['max_len'],
                roberta=(model_name == 'roberta_large')
            )

            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=config['tokenizer'])

            # Define training arguments (use unique output directory per fold)
            training_args = TrainingArguments(
                output_dir=f'./models/{model_name}',
                run_name=f"{model_name}_fold_{fold + 1}",
                overwrite_output_dir=True,
                eval_strategy='epoch',
                save_strategy='epoch',
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=32,
                num_train_epochs=5,
                warmup_steps=500,
                weight_decay=0.01,
                load_best_model_at_end=True,
                report_to='none'
            )

            # Initialize Trainer
            trainer = WeightedLossTrainer(
                model=copy.deepcopy(config['model']),
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                class_weights=class_weights_list
            )

            # Train and evaluate on the fold
            trainer.train()
            results = trainer.evaluate(val_dataset)

            # Record the fold results
            fold_metrics.append(results)
            print(f"Results for fold {fold + 1}: {results}")

        # Calculate average metrics across folds
        avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0].keys()}
        cv_results[model_name] = avg_metrics

        print(f"Average results for {model_name} across 3 folds: {avg_metrics}")

    # Display cross-validation results for all models
    print("\nCross-validation results for each model:")
    for model_name, metrics in cv_results.items():
        print(f"{model_name}: {metrics}")

    cv_results_df = pd.DataFrame.from_dict(cv_results, orient='index')
    cv_results_df.to_csv('./cross_validation_results.csv', index_label='Model')


if __name__ == '__main__':
    main()
