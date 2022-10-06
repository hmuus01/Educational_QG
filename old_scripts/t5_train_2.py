from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments


dataset_file = 'your/dataset/path.csv'
dataset = load_dataset('csv', data_files=dataset_file, split='train')

dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
val_dataset = dataset['test']

tokenizer = AutoTokenizer.from_pretrained('t5-base')


def tokenize(batch):
    tokenized_input = tokenizer(batch['source'], padding='max_length', truncation=True, max_length=max_source)
    tokenized_label = tokenizer(batch['target'], padding='max_length', truncation=True, max_length=max_target)

    tokenized_input['labels'] = tokenized_label['input_ids']

    return tokenized_input


train_dataset = train_dataset.map(tokenize, batched=True, batch_size=512)
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))

train_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('numpy', columns=['input_ids', 'attention_mask', 'labels'])

train_dataset.save_to_disk('your/train/save/dir')
val_dataset.save_to_disk('your/val/save/dir')


df = pd.read_csv(dataset_file)

source_text = df['source']
target_text = df['target']

tokenized_source_text = tokenizer(list(source_text), truncation=False, padding=False)
tokenized_target_text = tokenizer(list(target_text), truncation=False, padding=False)

max_source = 0
for item in tokenized_source_text['input_ids']:
    if len(item) > max_source:
        max_source = len(item)

max_target = 0
for item in tokenized_target_text['input_ids']:
    if len(item) > max_target:
        max_target = len(item)


model = T5ForConditionalGeneration.from_pretrained('t5-base')

output_dir = 'your/output/dir'

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=1, # Number of eval steps to keep in GPU (the higher, the mor vRAM used)
    prediction_loss_only=True, # If I need co compute only loss and not other metrics, setting this to true will use less RAM
    learning_rate=0.001,
    evaluation_strategy='steps', # Run evaluation every eval_steps
    save_steps=1000, # How often to save a checkpoint
    save_total_limit=1, # Number of maximum checkpoints to save
    remove_unused_columns=True, # Removes useless columns from the dataset
    run_name='run_name', # Wandb run name
    logging_steps=1000, # How often to log loss to wandb
    eval_steps=1000, # How often to run evaluation on the val_set
    logging_first_step=False, # Whether to log also the very first training step to wandb
    load_best_model_at_end=True, # Whether to load the best model found at each evaluation.
    metric_for_best_model="loss", # Use loss to evaluate best model.
    greater_is_better=False # Best model is the one with the lowest loss, not highest.
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
trainer.save_model(output_dir + '/model')