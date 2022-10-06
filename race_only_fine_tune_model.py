# Import packages
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# T5TokenizerFast
from transformers import ( AutoTokenizer as T5Tokenizer, AutoConfig, FlaxT5ForConditionalGeneration, AutoTokenizer
)

from qgmodel import QGModel
from qgmodel_t import QGModel_t

pl.seed_everything(42)

race_train_df = pd.read_csv('data/race/race_train_df.csv')
race_test_df = pd.read_csv('data/race/race_test_df.csv')
print(race_train_df.shape)

race_train_df.head()

race_dev_df = pd.read_csv('data/race/race_dev_df.csv')
print(race_dev_df.shape)

race_dev_df.head()

#Using paragraph
# context_name = 'context_para'
# drop_context = 'context_sent'

# df = sciq_squad_train_df.copy()
# print(df.shape, ' :copy')

# df = df.dropna() # One missing answer_text. Will fix it later.
# print(df.shape, ' :drop na')

#Dropping duplicates
# df = df.drop_duplicates(subset=['context_sent']).reset_index(drop=True)
# print(df.shape, ' :dropping duplicate sentence')

# df.rename(columns = {context_name: 'context'}, inplace=True)
# df.drop(columns=[drop_context, 'answer_start', 'answer_end'], inplace=True) #answer_start and answer_end are not needed and are for the paragraph
# print(df.shape, ' :final')

test_df = race_test_df
train_df = race_train_df

## Dev set
dev_df = race_dev_df
# dev_df.rename(columns = {context_name: 'context'}, inplace=True)
# dev_df.drop(columns=[drop_context, 'answer_start', 'answer_end'], inplace=True)

print(train_df.shape, 'train_df')
print(dev_df.shape, 'dev_df')
print(test_df.shape, 'test_df')

train_df.head()

SEP_TOKEN = '<sep>'
MASKING_CHANCE = 0.3 #30% chance to replace the answer with '[MASK]'

class QGDataset(Dataset):

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5Tokenizer,
            source_max_token_len: int,
            target_max_token_len: int
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        if np.random.rand() > MASKING_CHANCE:
            answer = data_row['correct']
        else:
            answer = '[MASK]'

        source_encoding = tokenizer(
            '{} {} {}'.format(answer, SEP_TOKEN, data_row['context']),
            max_length= self.source_max_token_len,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        target_encoding = tokenizer(
            '{} {} {}'.format(data_row['correct'], SEP_TOKEN, data_row['question']),
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            answer_text = data_row['correct'],
            context = data_row['context'],
            question = data_row['question'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )

class QGDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size,
        source_max_token_len: int,
        target_max_token_len: int
    ):

        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage=None):
        self.train_dataset = QGDataset(self.train_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
        self.val_dataset = QGDataset(self.val_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
        self.test_dataset = QGDataset(self.test_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers = 24)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=24)

MODEL_NAME = 't5-small'
# MODEL_NAME = 't5-base'
SOURCE_MAX_TOKEN_LEN = 300 #300
TARGET_MAX_TOKEN_LEN = 80

N_EPOCHS = 12
BATCH_SIZE = 8
LEARNING_RATE = 0.0001

DF_TAKE_PERCENTAGE = 1

TAKE_TRAIN = int(len(train_df) * DF_TAKE_PERCENTAGE)
TAKE_DEV = int(len(dev_df) * DF_TAKE_PERCENTAGE)
TAKE_TEST = int(len(test_df) * DF_TAKE_PERCENTAGE)

print('Taking', DF_TAKE_PERCENTAGE * 100, '%')
print(TAKE_TRAIN, 'of', len(train_df))
print(TAKE_DEV, 'of', len(dev_df))
print(TAKE_TEST, 'of', len(test_df))

print(train_df[:TAKE_TRAIN].shape, dev_df[:TAKE_DEV].shape, test_df[:TAKE_TEST].shape)
# print("MODEL: " +str(MODEL_NAME))
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_t5_pre') # Science small

# tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_full_t5_pre')
#tokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_5_more_epoch_version_2')
# tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_t5_pre')
print('tokenizer len before: ', len(tokenizer))
tokenizer.add_tokens(SEP_TOKEN)
print('tokenizer len after: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)


data_module = QGDataModule(train_df[:TAKE_TRAIN], dev_df[:TAKE_DEV], test_df[:TAKE_TEST], tokenizer, BATCH_SIZE, SOURCE_MAX_TOKEN_LEN, TARGET_MAX_TOKEN_LEN)
data_module.setup()

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='sci_small-fine_tuned_on_RACE_ft-test-best-checkpoint',
    save_top_k=-1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(
    checkpoint_callback= checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1, #TODO change change to 1
    progress_bar_refresh_rate=30
)
model = QGModel_t()
# model = QGModel()
# model = QGModel.load_from_checkpoint('checkpoints/best-checkpoint-v8.ckpt')


trainer.fit(model, data_module)
trainer.test()