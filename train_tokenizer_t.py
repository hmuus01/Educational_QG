import datasets

from t5_tokenizer_model import SentencePieceUnigramTokenizer
from transformers import T5Config
from t5_tokenizer_model import SentencePieceUnigramTokenizer as tok
from conf import dir

vocab_size = 64000


# Initialize a dataset
dataset = datasets.load_dataset('data/science_full', name="t_tok_pre", split="train")
input_sentence_size=len(dataset)
tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset)
    batch_length = 10
    for i in range(0, input_sentence_size, batch_length):
        yield dataset[i: i + batch_length]["text"]


# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

# Save files to disk
tokenizer.save("./"+dir+"/tokenizer.json")

# config = T5Config.from_pretrained("google/t5-v1_1-base", vocab_size=tokenizer.get_vocab_size())
config = T5Config.from_pretrained("google/t5-v1_1-small", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("./"+dir)