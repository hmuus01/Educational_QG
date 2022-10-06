import numpy
from transformers import T5TokenizerFast as T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer

from config import CONTEXT, QUESTION, ANSWER, SEP_TOKEN, SOURCE_MAX_TOKEN_LEN, TARGET_MAX_TOKEN_LEN

science_Atokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_t5_pre')
science_model = T5ForConditionalGeneration.from_pretrained("./t_from_checkpoint_science_t5_pre/", from_flax=True, return_dict=True)

def generate_science(qgmodel: science_model, tokenizer: science_Atokenizer, answer: str, context: str):
    source_encoding = tokenizer(
        '{} {} {}'.format(answer, SEP_TOKEN, context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    stat, words, counter = unknown_tags(answer, context, source_encoding, tokenizer)
    return ''.join(preds), stat, words, counter

# science_full_Atokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_science_full_t5_pre')
# science_full_model = T5ForConditionalGeneration.from_pretrained("./t_from_checkpoint_science_full_t5_pre/", from_flax=True, return_dict=True)

# science_full_Atokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_5')
# science_full_model = T5ForConditionalGeneration.from_pretrained("./pre_train_latest_5/", from_flax=True, return_dict=True)

science_full_Atokenizer = AutoTokenizer.from_pretrained('./pre_train_latest_5_more_epoch_version_2')
science_full_model = T5ForConditionalGeneration.from_pretrained("./pre_train_latest_5_more_epoch_version_2/", from_flax=True, return_dict=True)

#science_full_Atokenizer = AutoTokenizer.from_pretrained('./pre_train_on_t5_small_v2')
#science_full_model = T5ForConditionalGeneration.from_pretrained("./pre_train_on_t5_small_v2/",from_flax=True, return_dict=True)
def generate_science_full(answer, context, qgmodel=science_full_model, tokenizer=science_full_Atokenizer):
    # qgmodel: science_full_model, tokenizer: science_full_Atokenizer, answer: str, context: str
    # print('generate sci')

    source_encoding = tokenizer(
        '{} {} {}'.format(answer, SEP_TOKEN, context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    # stat, words = unknown_tags(answer, context, source_encoding, tokenizer)

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    stat, words, counter = unknown_tags(answer, context, source_encoding, tokenizer)
    return ''.join(preds), stat, words, counter


def unknown_tags(answer, context, source_encoding, tokenizer):
    stat = 0
    words = []
    source_np = source_encoding['input_ids'].numpy().flatten()
    ids = list(source_np)
    context_new = '{} {} {}'.format(answer, SEP_TOKEN, context)
    word_toks = tokenizer.tokenize(context_new)
    n_words = len(context.split())
    count = 0
    if 2 in ids:
        # context_new = '{} {} {}'.format(answer, SEP_TOKEN, context)
        # word_toks = tokenizer.tokenize(context_new)
        # n_words = len(context.split())
        # counter = 0

        for i, num in enumerate(ids):
            if num == 2:
                words.append(word_toks[i])
                count += 1

        stat += (count / n_words)
    return stat, words, count


cs_Atokenizer = AutoTokenizer.from_pretrained('./t_from_checkpoint_t5_pre')
cs_model = T5ForConditionalGeneration.from_pretrained("./t_from_checkpoint_t5_pre", from_flax=True, return_dict=True)

def generate_cs(qgmodel: cs_model, tokenizer: cs_Atokenizer, answer: str, context: str):
    source_encoding = tokenizer(
        '{} {} {}'.format(answer, SEP_TOKEN, context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    stat, words, count = unknown_tags(answer, context, source_encoding, tokenizer)
    return ''.join(preds), stat, words, count


def generate(qgmodel: T5ForConditionalGeneration, tokenizer: T5Tokenizer, answer: str, context: str):
    source_encoding = tokenizer(
        '{} {} {}'.format(answer, SEP_TOKEN, context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )


    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }
    stat, words, count = unknown_tags(answer, context, source_encoding, tokenizer)
    return ''.join(preds), stat, words, count


def show_result(generated: str, answer: str, context:str, original_question: str = ''):
    print('Generated: ', generated)
    if original_question:
        print('Original : ', original_question)

    print()
    print('Answer: ', answer)
    print('Conext: ', context)
    print('-----------------------------')