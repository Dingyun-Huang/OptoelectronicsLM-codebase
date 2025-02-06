from datasets import load_dataset
from transformers import AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizerFast, TrainingArguments, Trainer, DataCollatorWithPadding, RobertaConfig
import torch
from pprint import pprint
import os
import evaluate
import numpy as np
import wandb


accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=1)
    # print(predictions[:3])
    labels = np.argmax(labels, axis=1)
    # print(labels[:3])
    return accuracy.compute(predictions=predictions, references=labels)

model_path = "model_name_or_path"
tokenizer = AutoTokenizer.from_pretrained("model_name_or_path")

def preprocess_function(examples):
    
    abstr = [a.strip() for a in examples["abstract"]]
    title = [t.strip() for t in examples["title"]]
    temp = [title[i] + abstr[i] for i in  range(len(abstr))]
    inputs = tokenizer(
        temp,
        max_length=512,
        truncation=True,
        padding="max_length",
    )

    labels = []
    for label in examples["label"]:
        one_hot = [0.0] * 3
        one_hot[label] = 1.0
        labels.append(one_hot)

    inputs["label"] = labels
    return inputs


if __name__ == "__main__":
    
    id2label = {0: "LightEmitting", 1: "LightHarvesting", 2: "Photocatalysis"}
    label2id = {"LightEmitting": 0, "LightHarvesting": 1, "Photocatalysis": 2}
    for i in range(5):
        wandb.init(
            project="crossval ehc abstract with title",
            name=f"roberta-base-fold{i+1}",
            group="roberta-base-fix-case",
        )
        # change the dataset paths to your own dataset
        dataset_path = "Dingyun-Huang/emit-harvest-catalysis"
        train_d = load_dataset(dataset_path, split=f"train[:{i * 10}%]+train[{i * 10 + 10}%:]", delimiter="|")
        test_d = load_dataset(dataset_path, split=f"train[{i * 10}%:{i * 10 + 10}%]", delimiter="|")
        test_d = test_d.map(preprocess_function, batched=True,)
        train_d = train_d.map(preprocess_function, batched=True,)

        model = RobertaForSequenceClassification.from_pretrained("roberta-base", output_hidden_states=True, num_labels=3, id2label=id2label, label2id=label2id, 
                                                                problem_type="multi_label_classification")
        model.load_state_dict(torch.load(model_path), strict=False)
    
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
        training_args = TrainingArguments(
        output_dir="./roberta-base-ehc-crossval",
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0,
        eval_accumulation_steps=1,
        # run_name=f"roberta-base-fold{i+1}"
        # load_best_model_at_end=True,
        )
    
        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_d,
        eval_dataset=test_d,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        )
    
        trainer.train()
        wandb.finish()



