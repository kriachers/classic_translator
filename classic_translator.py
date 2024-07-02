from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline
import pandas as pd
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import os

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

"""
READ DATASET
"""

current_directory = os.path.dirname(os.path.abspath(__file__))
parallel_file = os.path.join(current_directory, 'texts', 'lady_susan.txt')
eng_file = os.path.join(current_directory, 'texts', 'lady_susan_en.txt')
ru_file = os.path.join(current_directory, 'texts', 'lady_susan_ru.txt')

def split_parallel_file(parallel_file, eng_file, ru_file):
    with open(parallel_file, 'r', encoding='utf-8') as infile, \
         open(eng_file, 'w', encoding='utf-8') as eng_outfile, \
         open(ru_file, 'w', encoding='utf-8') as ru_outfile:

        lines = infile.readlines()

        i = 0
        while i < len(lines):
            eng = lines[i].strip()
            if i + 1 < len(lines):
                ru = lines[i + 1].strip()
            else:
                ru = ''
            if eng and ru:
                eng_outfile.write(eng + '\n')
                ru_outfile.write(ru + '\n')
            i += 2  # Adjust this to 2 to ensure we move correctly through the lines

# Split
split_parallel_file(parallel_file, eng_file, ru_file)

print(f"English sentences saved to: {eng_file}")
print(f"Russian sentences saved to: {ru_file}")

"""
PARALLEL DATADICT
"""


import pandas as pd

# Step 1: Read the parallel file
with open(parallel_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Step 2: Parse the content
en_sentences = []
ru_sentences = []
for i in range(0, len(lines), 2):  # Assuming English and Russian lines are alternating
    en_sentences.append(lines[i].strip())
    if i+1 < len(lines):
        ru_sentences.append(lines[i+1].strip())

# Step 3: Create a DataFrame
data = {'en': en_sentences, 'ru': ru_sentences}
df = pd.DataFrame(data)

# Step 4: Convert to DatasetDict
dataset = Dataset.from_pandas(df)
dataset_dict = DatasetDict({'train': dataset})

# Print the dataset_dict to verify
print(dataset_dict)

"ADD VALID AND TEST"

test_ratio = 0.10
validation_ratio = 0.05

# Calculate the number of rows for each subset
num_rows = len(df)
test_size = int(num_rows * test_ratio)
validation_size = int(num_rows * validation_ratio)
train_size = num_rows - test_size - validation_size

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the DataFrame
train_df = df[:train_size]
validation_df = df[train_size:train_size + validation_size]
test_df = df[train_size + validation_size:]

# Convert DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

# Print the dataset_dict to verify
print(dataset_dict)

"Load the T5 tokenizer"

model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

source_lang = "ru"
target_lang = "en"
prefix = "translate Russian to English: "

def preprocess_function(examples):
    inputs = [prefix + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_data = dataset_dict.map(preprocess_function, batched=True)

print(tokenized_data['train'][10])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    result = {k: round(v, 4) for k, v in result.items()}
    return result

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    save_total_limit=3,
    num_train_epochs=2,
    use_mps_device=True

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model('T5_checkpoint')
