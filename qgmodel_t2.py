import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AdamW, T5TokenizerFast, FlaxT5ForConditionalGeneration, \
    T5Tokenizer, AutoTokenizer

from config import SEP_TOKEN

# pre-train >>> fine-tune
#

class QGModel_t2(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # MODEL_NAME = 't5-small'
        # MODEL_NAME = 'qg-t5-new'
        tokenizer = AutoTokenizer.from_pretrained('./pre_train_on_t5_small_v2')
        # tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.add_tokens(SEP_TOKEN)
        TOKENIZER_LEN=len(tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained("./pre_train_on_t5_small_v2/",from_flax=True, return_dict=True) # T5 small Science
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