import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evaluate
import numpy as np
import pandas as pd


from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    default_data_collator,
    set_seed
)
from datasets import (
    load_dataset
)


from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


set_seed(42)

model_name = "bert-base-uncased"

n_epoch = 3
n_batch_size = 32

metric = evaluate.load("accuracy", cache_dir="outputs/")

training_arguments = TrainingArguments(
    num_train_epochs=n_epoch,
    output_dir="output/",
    per_device_train_batch_size=n_batch_size,
    per_device_eval_batch_size=n_batch_size,
    # NOTE: learning rate defaults to 5e-5 if not explicitly set
    learning_rate=1e-5,
    # NOTE: fp16 requires installing apex
    fp16=False,
    # logging
    # logging_strategy="steps",
    logging_steps=10,
    disable_tqdm=False,
    # checkpoint setting
    # save_strategy="steps",
    save_steps=100,
    # evaluation
    evaluation_strategy="steps",
    eval_steps=100,
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

def tokenize_dataset(record):
    result = tokenizer(record["sentence"], padding="max_length", truncation=True)
    result["label"] = record["label"]

    return result

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

dataset = load_dataset("sst2")
tokenized_dataset = dataset.map(
    tokenize_dataset,
    batched=True,
    num_proc=None,
    load_from_cache_file=False,
)

removed_columns = [
    "idx",
    "sentence"
]

train_dataset = tokenized_dataset["train"].remove_columns(removed_columns)
eval_dataset = tokenized_dataset["validation"].remove_columns(removed_columns)

# sampling
train_dataset = train_dataset.select(np.random.choice(range(train_dataset.num_rows), 5000))

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=default_data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

predict_dataset = eval_dataset.remove_columns("label")
predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions

y_true = eval_dataset["label"]
y_pred = np.argmax(predictions, axis=1)

df = pd.DataFrame(eval_dataset)
df["prediction"] = y_pred

df.to_pickle("/home/gyang16/test_slurm/result.pkl")

#               precision    recall  f1-score   support
#
#            0       0.88      0.86      0.87       428
#            1       0.87      0.89      0.88       444
#
#     accuracy                           0.88       872
#    macro avg       0.88      0.87      0.87       872
# weighted avg       0.88      0.88      0.87       872

