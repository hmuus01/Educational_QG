from os import path
CONTEXT = 'context'
QUESTION = 'question'
ANSWER = 'answer'

SEP_TOKEN = '<sep>'
CLS_TOKEN = '[CLS]'
MASK='[MASK]'

MASKING_CHANCE = 0.3  # 30% chance to replace the answer with '[MASK]'
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
N_EPOCHS = 5
BATCH_SIZE = 16
DF_TAKE_PERCENTAGE = 1
NUM_WORKERS=2

SQUAD_N_TEST = 11877
RACE_N_TEST = 11877

SQUAD_TRAIN = path.join('data', 'squad', 'train_df.csv')

SQUAD_DEV = path.join('data', 'squad', 'dev_df.csv')

RACE_TRAIN = path.join('data', 'race', 'race_train_df.csv')

RACE_DEV = path.join('data', 'race', 'race_dev_df.csv')

SCIQ_DEV = path.join('data', 'sciq', 'test.csv')

CHECKPOINT_PATH_V9 = path.join('checkpoints', 'best-checkpoint-v9.ckpt')

# CHECKPOINT_PATH_V9 = path.join('checkpoints', 'multitask-qg-ag.ckpt')

# CHECKPOINT_PATH_V9 = path.join('checkpoints', 'best-checkpoint.ckpt')

