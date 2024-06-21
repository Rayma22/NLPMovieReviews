from transformers import DebertaTokenizer, DebertaModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

class DebertaWithDenseLayers(nn.Module):
    """
    Custom DeBERTa model with additional dense layers for text classification.
    """
    def __init__(self, num_labels):
        super(DebertaWithDenseLayers, self).__init__()
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.dense1 = nn.Linear(self.deberta.config.hidden_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dense1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.dense2(x)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.dense2.out_features), labels.view(-1))
            return loss, logits
        else:
            return logits

def compute_metrics(pred):
    """
    Computes the macro F1 score from the model's predictions.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {"eval_f1": f1_score(labels, preds, average='macro')}


def main():
    """
    Trains and evaluates the DeBERTa model with additional dense layers on the Rotten Tomatoes dataset.
    """
    # Load and preprocess the dataset
    dataset = load_dataset("rotten_tomatoes")
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Initialize the custom model
    model = DebertaWithDenseLayers(num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end= True
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test dataset to get the metrics for the last epoch
    eval_results = trainer.evaluate()

    # Prediction on the test dataset
    predictions = trainer.predict(tokenized_datasets["test"])
    preds = np.argmax(predictions.predictions, axis=1)

    # Extract the macro F1 score and print it
    macro_f1_score = eval_results["eval_f1"]
    print(f" {macro_f1_score:.3f}")

    # Save predictions to CSV
    
    df = pd.DataFrame({
        'index': range(len(preds)),
        'pred': preds
    })
    df.to_csv('results.csv', index=False)

if __name__ == "__main__":
    main()