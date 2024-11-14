
from transformers import TrainingArguments
import datasets
dataset  = datasets.load_dataset("tweet_eval", name="sentiment")
training_args = TrainingArguments(output_dir="test_trainer")
from transformers import RobertaForSequenceClassification, RobertaTokenizer

model_name = 'roberta-base'
num_labels = 7

model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


#tokenizing data

def tokenize_function(examples):
    return tokenizer(examples["text"], padding= "max_length", truncation=True)

encoded_dataset = dataset.map(tokenize_function, batched=True)


#fine tuning

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= encoded_dataset["train"],
    eval_dataset= encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)



import torch
import time

# Before training
print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()} bytes")
print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved()} bytes")
start_time = time.time()

trainer.train()

# After training
end_time = time.time()
print(f"Final GPU memory allocated: {torch.cuda.memory_allocated()} bytes")
print(f"Final GPU memory reserved: {torch.cuda.memory_reserved()} bytes")


print(f"Train Time taken: {end_time - start_time} seconds")
