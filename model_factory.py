
# Import packages
from config import SEP_TOKEN
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer
from qgmodel import QGModel as QG
from qgmodel_t import QGModel_t as QG_t
from qgmodel_t2 import QGModel_t2 as QG_t2
from config import CHECKPOINT_PATH_V9

class T5():
    def __init__(self):
        MODEL_NAME = 't5-small'
        self.__tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.__tokenizer.add_tokens(SEP_TOKEN)
        self.__model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.__model.resize_token_embeddings(len(self.__tokenizer))  # resizing after adding new tokens to the tokenizer

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def model(self):
        return self.__model


class ModelFactory():
    def get(self, name):
        if name == 'leaf':
            MODEL_NAME = 't5-small'
            tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
            tokenizer.add_tokens(SEP_TOKEN)
            print('model from checkpoint\n')
            # checkpoint_path = CHECKPOINT_PATH_V9
            # checkpoint_path = 'checkpoints/leaf-fine_tuned_on_sciq_ft-test-best-checkpoint.ckpt'
            checkpoint_path = 'checkpoints/leaf-fine_tuned_on_RACE_ft-test-best-checkpoint-v5.ckpt'

            model = QG.load_from_checkpoint(checkpoint_path)
            print('model freeze\n')
            model.freeze()
            print('model eval\n')
            model.eval()
            return model.model, tokenizer

        if name == 'T5':
            model_object = T5()
            return model_object.model, model_object.tokenizer

        if name == 'cs':
            # tokenizer = AutoTokenizer.from_pretrained('./t_t5_pre/') way old
            tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_t5_pre')

            tokenizer.add_tokens(SEP_TOKEN)

            print('model from checkpoint\n')
            #checkpoint_path = 'checkpoints/cs-test-best-checkpoint-v3.ckpt'
            # checkpoint_path = 'checkpoints/cs-fine_tuned_on_sciq_ft-test-best-checkpoint-v3.ckpt'
            checkpoint_path = 'checkpoints/cs-fine_tuned_on_RACE_ft-test-best-checkpoint-v5.ckpt'

            model = QG_t.load_from_checkpoint(checkpoint_path)
            print('model freeze\n')
            model.freeze()
            print('model eval\n')
            model.eval()
            return model.model, tokenizer


        if name == 'sci':
            # tokenizer = AutoTokenizer.from_pretrained('./t_t5_pre/')
            tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_t5_pre')
            tokenizer.add_tokens(SEP_TOKEN)

            print('model from checkpoint\n')
            # checkpoint_path = 'checkpoints/test-best-checkpoint-v4.ckpt'
            checkpoint_path = 'checkpoints/science-test-best-checkpoint-v3.ckpt'
            # checkpoint_path = 'checkpoints/sci_small-fine_tuned_on_sciq_ft-test-best-checkpoint-v4.ckpt'
            # checkpoint_path = 'checkpoints/sci_small-fine_tuned_on_RACE_ft-test-best-checkpoint-v4.ckpt'



            model = QG_t.load_from_checkpoint(checkpoint_path)
            print('model freeze\n')
            model.freeze()
            print('model eval\n')
            model.eval()
            return model.model, tokenizer

        if name == 'sci_full':
            tokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_full_t5_pre')
            # MODEL_NAME = 't5-small'
            # tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
            # tokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_with_new_tok_small_v3')

            tokenizer.add_tokens(SEP_TOKEN)

            print('model from checkpoint\n')
            # checkpoint_path = 'checkpoints/science-full-test-best-checkpoint-v4.ckpt'
            # checkpoint_path = 'checkpoints/science_sci_squad-test-best-checkpoint-v3.ckpt'

            #checkpoint_path = 'checkpoints/science-5-epoch-best-checkpoint-v4.ckpt'

            # checkpoint_path = 'checkpoints/pre-t5_small_science-best-checkpoint-v6.ckpt'
            # checkpoint_path = 'checkpoints/fine_tuned_on_sciq_ft-test-best-checkpoint-v6.ckpt'
            #checkpoint_path = 'checkpoints/Sci-fine_tuned_on_sciq_ft-test-best-checkpoint-v6.ckpt'

            checkpoint_path = 'checkpoints/sci-fine_tuned_on_RACE_ft-test-best-checkpoint-v14.ckpt'
            # checkpoint_path = 'checkpoints/t5_small_science-best-checkpoint-v3.ckpt'
            # checkpoint_path = "checkpoints/t5_small_science-best-checkpoint-v39.ckpt"
            # checkpoint_path = 'checkpoints/best-checkpoint.ckpt'
            #model = QG_t2.load_from_checkpoint(checkpoint_path)
            model = QG_t.load_from_checkpoint(checkpoint_path)
            # model = QG.load_from_checkpoint(checkpoint_path)
            print('model freeze\n')
            model.freeze()
            print('model eval\n')
            model.eval()
            return model.model, tokenizer
        if name == '23':
            return
        if name == '3':
            return

