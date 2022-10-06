import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AdamW, T5TokenizerFast, FlaxT5ForConditionalGeneration, \
    T5Tokenizer, AutoTokenizer

from config import SEP_TOKEN

# pre-train >>> fine-tune
#

class QGModel_t(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # MODEL_NAME = 't5-small'
        # MODEL_NAME = 'qg-t5-new'
        tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_t5_pre')
        # tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_t5_pre') #Computer Science
        #tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_full_t5_pre')
        #tokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_5') #SCIENCE FULL 5 EPOCHS
        # tokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_5_more_epoch_version_2') #SCIENCE FULL 5 EPOCHS MORE
        #tokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_with_new_tok_small_v3') #pre trained from scratch
        # tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        # tokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_with_new_tok_small_v3')
        tokenizer.add_tokens(SEP_TOKEN)
        TOKENIZER_LEN=len(tokenizer)
        # self.model = FlaxT5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        # self.model = T5ForConditionalGeneration.from_pretrained("./t_t5_pre/", from_flax=True, return_dict=True)
        self.model = T5ForConditionalGeneration.from_pretrained("./t_from_checkpoint_science_t5_pre/", from_flax=True, return_dict=True) # SCIENCE
        # self.model = T5ForConditionalGeneration.from_pretrained("./t_from_checkpoint_t5_pre/", from_flax=True, return_dict=True) # COMPUTER SCIENCE
        # self.model = T5ForConditionalGeneration.from_pretrained("./t_from_checkpoint_science_full_t5_pre/", from_flax=True, return_dict=True) # SCIENCE FULL
        #self.model = T5ForConditionalGeneration.from_pretrained("./pre_train_latest_5/", from_flax=True, return_dict=True) #SCIENCE FULL 5 EPOCHS
        # self.model = T5ForConditionalGeneration.from_pretrained("./pre_train_latest_5_more_epoch_version_2/", from_flax=True, return_dict=True) #SCIENCE FULL 5 EPOCHS MORE
        #self.model = T5ForConditionalGeneration.from_pretrained("./pre_train_latest_with_new_tok_small_v3/", from_flax=True, return_dict=True)
        # self.model = T5ForConditionalGeneration.from_pretrained("./pre_train_latest_with_new_tok_small_v3/", from_flax=True, return_dict=True) #NEW pretsined scratch

        self.model.resize_token_embeddings(TOKENIZER_LEN)  # resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        LEARNING_RATE=0.0001
        return AdamW(self.parameters(), lr=LEARNING_RATE)