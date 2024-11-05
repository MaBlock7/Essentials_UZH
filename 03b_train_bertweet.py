import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer,
                          EarlyStoppingCallback)


def prevent_data_slippage(train_df, test_df, col_name='message'):
    original_len = len(test_df)
    test_df = test_df[~(test_df[col_name].isin(train_df[col_name]))]
    print(f'Removed {original_len - len(test_df)} entries from the test set!')
    return train_df, test_df

def tokenize_data(texts, tokenizer, max_len):
    return tokenizer(texts, truncation=True, padding=True, max_length=max_len)

class MessagesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenize_data(texts, tokenizer, max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    f1_average = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    return {
        'f1 weighted average': f1_average,
        'precision': precision,
        'recall': recall,
    }

def main():
    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    model_name = 'bertweet'

    # Read in the training data
    df = pd.read_csv('./training_data_no_duplicates_per_channel.csv')
    df = df.dropna(subset=['message'])
    print(f"Total samples: {len(df)}")

    # Calculate the class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    class_weights_list = [class_weights_dict[i] for i in range(len(class_weights_dict))]
    print(f"Class weights: {class_weights_dict}")

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Remove any data leakage between train and val
    train_df, val_df = prevent_data_slippage(train_df, val_df)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        'vinai/bertweet-base',
        num_labels=6,
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = MessagesDataset(
        train_df['message'].tolist(),
        train_df['label'].astype(int).tolist(),
        tokenizer,
        128
    )

    val_dataset = MessagesDataset(
        val_df['message'].tolist(),
        val_df['label'].astype(int).tolist(),
        tokenizer,
        128
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tune_hyperparams = False
    if tune_hyperparams:
        # Define initial training arguments
        training_args = TrainingArguments(
            output_dir=f'./models/{model_name}',
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
            metric_for_best_model='eval_f1 weighted average',
            greater_is_better=True,
            report_to='none',
            seed=42,
            fp16=True,
            logging_steps=50,
        )

        # Hyperparameter search space
        def hp_space(trial):
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
                'num_train_epochs': trial.suggest_int('num_train_epochs', 3, 6),
                'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [8, 16]),
                'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.3),
                'warmup_steps': trial.suggest_int('warmup_steps', 0, 1000),
            }

        # Model initialization function
        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(
                'vinai/bertweet-base',
                num_labels=6,
                problem_type="single_label_classification"
            )

        # Initialize Trainer with hyperparameter search
        trainer = WeightedLossTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights_list
        )

        # Hyperparameter search
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            hp_space=hp_space,
            backend="optuna",
            n_trials=40,
            compute_objective=lambda metrics: metrics['eval_f1 weighted average']
        )

        print(f"Best trial hyperparameters: {best_trial.hyperparameters}")

        # Update training arguments with best hyperparameters
        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

        # Re-initialize the trainer with the best hyperparameters
        trainer = WeightedLossTrainer(
            model_init=model_init,
            args=trainer.args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights=class_weights_list
        )

        # Train the final model on the training set
        trainer.train()

        # Evaluate on the validation set
        results = trainer.evaluate(val_dataset)
        print(f"Validation Results: {results}")

    # Retrain on the full dataset for production
    retrain_on_full_data = True
    if retrain_on_full_data:
        print("\nRetraining on the full dataset with the best hyperparameters...")

        # Combine training and validation data
        full_dataset = MessagesDataset(
            df['message'].tolist(),
            df['label'].astype(int).tolist(),
            tokenizer,
            128
        )

        full_training_args = TrainingArguments(
            output_dir=f'./models/{model_name}_full',
            overwrite_output_dir=True,
            evaluation_strategy='no',
            save_strategy='epoch',
            learning_rate=float(best_trial.hyperparameters['learning_rate']),
            per_device_train_batch_size=int(best_trial.hyperparameters['per_device_train_batch_size']),
            per_device_eval_batch_size=32,
            num_train_epochs=int(best_trial.hyperparameters['num_train_epochs']),
            warmup_steps=int(best_trial.hyperparameters['warmup_steps']),
            weight_decay=float(best_trial.hyperparameters['weight_decay']),
            report_to='none',
            seed=42,
            logging_steps=50,
            fp16=True
        )

        # Initialize a new Trainer for retraining
        retrain_trainer = Trainer(
            model=model,
            args=full_training_args,
            train_dataset=full_dataset,
            data_collator=data_collator,
        )

        # Train on full data
        retrain_trainer.train()

        # Save the final model
        retrain_trainer.save_model(f'./models/{model_name}_final')


if __name__ == '__main__':
    main()
