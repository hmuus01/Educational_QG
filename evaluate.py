# Import packages
import os
import re
import string
from collections import Counter

import pandas as pd
import torch
import language_tool_python
from rouge_score import rouge_scorer

from nltk import word_tokenize, ngrams
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from tqdm import tqdm
from data_factory import DataFactory
from model_factory import ModelFactory
from utils import generate, generate_science, generate_science_full, generate_cs
import config as cfg
from nltk.translate.bleu_score import SmoothingFunction
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer

model_bert = 'bert-base-uncased'
tool = language_tool_python.LanguageTool('en-US')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths_inspect(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths), ground_truths[np.argmax(scores_for_ground_truths)]

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        # print("ground truth " +str(ground_truth))
        # print("prediction " + str(prediction))
        score = metric_fn(prediction, ground_truth)

        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

############
############
def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())


def lexical_diversity(prediction, rank=3):

    n_grams = list(ngrams(prediction.split(), rank))
    distinct_ngrams = set(ngrams(prediction.split(), rank))

    return len(distinct_ngrams) / len(prediction.split())



def run_t5_on(dfin, modelin, model_pp, tokenizer_pp, tokenizer, model_name="leaf"):

    input_answer = '[MASK]'
    contexts = list(set(dfin['context']))
    predictions=[]
    perplexity= []
    divs = []
    gram = []
    for context in tqdm(contexts):
        num_words = len(context.split())
        for i in range(num_words//512):
            ctxt = context[i*512: (i+1)*512]
            print("Context "+str(ctxt))
            if model_name == "leaf":
                prediction, stat, words, count = generate(modelin, tokenizer, input_answer, ctxt)
            elif model_name == "cs":
                prediction, stat, words, count = generate_cs(modelin, tokenizer, input_answer, ctxt)
            elif model_name == "sci_full":
                prediction, stat, words, count = generate_science_full(input_answer, ctxt, modelin, tokenizer)
            else:
                prediction, stat, words, count = generate_science(modelin, tokenizer, input_answer, ctxt)


            if cfg.SEP_TOKEN in prediction:
                prediction = prediction.split(cfg.SEP_TOKEN)[1]
            predictions.append(prediction)
            # Perplexity
            ppl_score = score(sentence=ctxt, model=model_pp, tokenizer=tokenizer_pp)
            perplexity.append(ppl_score)
            # Diversity
            diversity = lexical_diversity(ctxt)
            divs.append(diversity)
            # Grammer
            matches = tool.check(ctxt)
            gram.append(len(matches))

    return predictions, perplexity, divs, gram

def evaluate_t5_on(dfin, modelin, tokenizer, model_pp, tokenizer_pp, model_name="leaf", n_samples=100):
    results = {
        'f1':[], 'bleu_1':[], 'bleu_2':[], 'bleu_3':[], 'bleu_4':[], 'ppl_scores':[], 'divs': [], 'grammer':[], 'stats':[], 'words':[], 'count':[]
    }
    input_answer = '[MASK]'
    contexts = list(set(dfin['context']))

    counter = 0

    rouges=[]
    predictions=[]

    for context in tqdm(contexts):
        if counter > n_samples > 0:
            continue

        context_df = dfin[dfin['context'] == context]

        gts = [ f'{row["question"]}' for i, row in context_df.iterrows()]

        if model_name == "leaf":
            prediction, stat, words, count = generate(modelin, tokenizer, input_answer, context)
        elif model_name == "cs":
            prediction, stat, words, count = generate_cs(modelin, tokenizer, input_answer, context)
        elif model_name == "sci_full":
            prediction, stat, words, count = generate_science_full(input_answer, context, modelin, tokenizer)
        else:
            prediction, stat, words, count = generate_science(modelin, tokenizer, input_answer, context)


        if cfg.SEP_TOKEN in prediction:
            prediction = prediction.split(cfg.SEP_TOKEN)[1]


        results['stats'].append(stat)
        results['words'].append(words)
        results['count'].append(count)


        # Calc f1
        score_f1 = metric_max_over_ground_truths(f1_score, prediction, gts)

        # Calc Diversity
        diversity = lexical_diversity(prediction)
        results['divs'].append(diversity)

        predictions.append(prediction)
        #
        # Calc Perplexity
        ppl_score = score(sentence=prediction, model=model_pp, tokenizer=tokenizer_pp)
        results['ppl_scores'].append(ppl_score)

        #Calc Grammer
        matches = tool.check(prediction)
        results['grammer'].append(len(matches))



        # # Calc Bleu
        gts_tokens = list(map(lambda x: word_tokenize(x), gts))
        prediction_tokens = word_tokenize(prediction)
        chencherry = SmoothingFunction()
        bleu_score1 = sentence_bleu(gts_tokens, prediction_tokens, weights=(1,0,0,0), smoothing_function=chencherry.method2)
        bleu_score2 = sentence_bleu(gts_tokens, prediction_tokens, weights=(1./2., 1./2.), smoothing_function=chencherry.method2)
        bleu_score3 = sentence_bleu(gts_tokens, prediction_tokens, weights=(1./3., 1./3., 1./3.), smoothing_function=chencherry.method2)
        bleu_score4 = sentence_bleu(gts_tokens, prediction_tokens, weights=(1./4., 1./4., 1./4., 1./4.), smoothing_function=chencherry.method2)
        # bleu_score = sentence_bleu(gts_tokens, prediction_tokens, weights=[ (1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.) ], smoothing_function=chencherry.method2)


        counter += 1

        results['f1'].append(score_f1)
        # results['bleu'].append(bleu_score)
        results['bleu_1'].append(bleu_score1)
        results['bleu_2'].append(bleu_score2)
        results['bleu_3'].append(bleu_score3)
        results['bleu_4'].append(bleu_score4)

        # rouges.append(rouge_score)

    return results, predictions

def inspect(dfin, modelin, tokenizer, indices, model_name="leaf"):
    input_answer = cfg.MASK
    contexts = list(set(dfin['context']))

    for i,context in enumerate(contexts):
        if i not in indices:
            continue
        # generate a question, answer pair
        # subset dev_df for ctxt=contxt
        context_df = dfin[dfin['context'] == context]
        # collect all ground_truths for ctxt and call metric_max_over_ground_truths
        #gts = [f'{row["answer"]} {cfg.SEP_TOKEN} {row["question"]}' for i, row in context_df.iterrows()]
        # gts = context_df[answer_header].map(lambda x: '{} {} {}'.format(x, SEP_TOKEN, context)).tolist()
        # gts_answer = [f'{row["answer"]}' for i, row in context_df.iterrows()]
        gts = [f'{row["question"]}' for i, row in context_df.iterrows()]
        # gts = context_df[answer_header].map(lambda x: '{} {} {}'.format(x, SEP_TOKEN, context)).tolist()

        if model_name == "leaf":
            prediction = generate(modelin, tokenizer, input_answer, context)
        elif model_name == "cs":
            prediction = generate_cs(modelin, tokenizer, input_answer, context)
        elif model_name == "sci_full":
            prediction = generate_science_full(modelin, tokenizer, input_answer, context)
        else:
            prediction = generate_science(modelin, tokenizer, input_answer, context)


        if cfg.SEP_TOKEN in prediction:
            prediction = prediction.split(cfg.SEP_TOKEN)[1]
        print('prediction:')
        print(prediction)
        score_f1, gt = metric_max_over_ground_truths_inspect(f1_score, prediction, gts)
        print('f1 match:')
        print(gt)
        # gts_tokens = list(map(lambda x: word_tokenize(x), gts))
        # prediction_tokens = word_tokenize(prediction)
        bleu_score, gt = metric_max_over_ground_truths_inspect(sentence_bleu, prediction, gts)
        print('bleu match:')
        print(gt)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')

def generate_question(modelin, tokenizer, model_name="leaf"):
    input_answer = cfg.MASK
    print("--------===================----------------")
    # context = "Errors in map-making tasks using computer vision are sparse. We demonstrate this by considering the construction of digital elevation models that employ stereo matching algorithms to triangulate real-world points. This sparsity, coupled with a geometric theory of errors recently developed by the authors, allows for autonomous agents to calculate their own precision independently of ground truth. We connect these developments with recent advances in the mathematics of sparse signal reconstruction or compressed sensing. " \
    #               "The theory presented here extends the autonomy of 3-D model reconstructions discovered in the 1990s to their errors."

    # context = "Muscles That Move the Head The head, attached to the top of the vertebral column, is balanced, moved, and rotated by the neck muscles (Table 11.5). When these muscles act unilaterally, the head rotates. When they contract bilaterally, the head flexes or extends. The major muscle that laterally flexe and rotates the head is the sternocleidomastoid. In addition, both muscles working together are the flexors of the head. Place your fingers on both sides of the neck and turn your head to the left and to the right. You will feel the movement originate there. This muscle divides the neck into anterior and posterior triangles when viewed from the side (Figure 11.14)"

    #context = "Biodiversity refers to the variety of life and its processes, including the variety of living organisms, the genetic differences among them, and the communities and ecosystems in which they occur. Scientists have identified about 1.9 million species alive today. They are divided into the six kingdoms of life shown in Figure below . Scientists are still discovering new species. Thus, they do not know for sure how many species really exist today. Most estimates range from 5 to 30 million species."

    #context = "Take-Home Experiment: The Pupil Look at the central transparent area of someone\u2019s eye, the pupil, in normal room light. Estimate the diameter of the pupil. Now turn off the lights and darken the room. After a few minutes turn on the lights and promptly estimate the diameter of the pupil. What happens to the pupil as the eye adjusts to the room light? Explain your observations. The eye can detect an impressive amount of detail, considering how small the image is on the retina. To get some idea of how small the image can be, consider the following example."

    context = "In both eukaryotes and prokaryotes, ribosomes are the non-membrane bound organelles where proteins are made. Ribosomes are like the machines in the factory that produce the factory's main product. Proteins are the main product of the cell."
    if model_name == "leaf":
        prediction = generate(modelin, tokenizer, input_answer, context)
    elif model_name == "cs":
        prediction = generate_cs(modelin, tokenizer, input_answer, context)
    elif model_name == "sci_full":
        prediction = generate_science_full(modelin, tokenizer, input_answer, context)
    else:
        prediction = generate_science(modelin, tokenizer, input_answer, context)

    if cfg.SEP_TOKEN in prediction:
        prediction = prediction.split(cfg.SEP_TOKEN)[1]

    print("context pred " +str(prediction))


def perplexity_dataset(dfin, model_pp, tokenizer_pp, n_samples=100):
    counter = 0


    contexts = list(set(dfin['context']))
    results = {
        'ppl_scores': [], 'divs': [], 'grammer': []
    }

    for context in tqdm(contexts):
        if counter > n_samples > 0:
            continue
        # generate a question, answer pair
        # subset dev_df for ctxt=contxt
        context_df = dfin[dfin['context'] == context]
        # collect all ground_truths for ctxt and call metric_max_over_ground_truths
        # ground_truths = list(map(lambda x: x['text'], qa['answers']))
        gts = [f'{row["question"]}' for i, row in context_df.iterrows()]
        for gt in gts:
            # print(gt)
            ppl_score = score(sentence=gt, model=model_pp, tokenizer=tokenizer_pp)
            results['ppl_scores'].append(ppl_score)
            matches = tool.check(gt)
            diversity = lexical_diversity(gt)
            results['divs'].append(diversity)
            results['grammer'].append(len(matches))
            counter += 1
    return results

if __name__ == '__main__':
    # TODO: move to command line params
    # model = 'leaf'
    # model = 'cs'
    model = 'sci'
    # model = 'sci_full'

    data_1 = 'squad'
    data_2 = 'race'
    data_3 = 'sciq'
    n_samples = 0
    debug=False
    print("--------------------------==============-==============-------------------")
    print("Model " + str(model))
    print("--------------------------==============-==============-------------------")
    data_factory = DataFactory()
    model_factory = ModelFactory()

    squa_dev_df = data_factory.get_dev(data_1)
    race_dev_df = data_factory.get_dev(data_2)
    sciq_dev_df = data_factory.get_dev(data_3)
    best_model, tokenizer = model_factory.get(model)

    model_pp = AutoModelForMaskedLM.from_pretrained(model_bert)
    tokenizer_pp = AutoTokenizer.from_pretrained(model_bert)


    ####################################################
    # generate_question(best_model, tokenizer, model_name="leaf")
    ####################################################

    ###################################################
    res = perplexity_dataset(sciq_dev_df, model_pp=model_pp, tokenizer_pp=tokenizer_pp, n_samples=n_samples)
    ppl_scores = res['ppl_scores']
    grammer = res['grammer']
    print(f'Mean PPL scores {data_3}: {np.mean(ppl_scores)}')
    print("Grammer Error " + str(np.mean(grammer)))
    divs = res['divs']
    print("Diversity " + str(np.mean(divs)))
    ###################################################
    ###################################################
    res3 = perplexity_dataset(race_dev_df, model_pp=model_pp, tokenizer_pp=tokenizer_pp, n_samples=n_samples)
    ppl_scores = res3['ppl_scores']
    grammer = res3['grammer']
    divs = res3['divs']
    print(f'Mean PPL scores {data_2}: {np.mean(ppl_scores)}')
    print("Grammer Error " + str(np.mean(grammer)))
    print("Diversity " + str(np.mean(divs)))
    ##################################################
    res2 = perplexity_dataset(squa_dev_df, model_pp=model_pp, tokenizer_pp=tokenizer_pp, n_samples=n_samples)
    ppl_scores = res2['ppl_scores']
    grammer = res2['grammer']
    divs = res2['divs']
    print(f'Mean PPL scores {data_1}: {np.mean(ppl_scores)}')
    print("Grammer Error " + str(np.mean(grammer)))
    print("Diversity " + str(np.mean(divs)))

    # path = os.path.join('data', 'cntxt.csv')
    # print('reading csv')
    # data_df = pd.read_csv(path, sep='\t')
    #
    # model_2 = 'leaf'
    #
    # best_model_2, tokenizer_2 = model_factory.get(model_2)
    # predictions, perplexity, divs, gram = run_t5_on(dfin=data_df, modelin=best_model_2, tokenizer=tokenizer_2, model_pp=model_pp, tokenizer_pp=tokenizer_pp, model_name=model_2)
    #
    # [print(x[:-4]) for x in predictions]
    # print(f'Mean PPL scores: {np.mean(perplexity)}')
    # print("Diversity " + str(np.mean(divs)))
    # print("Grammer Error " + str(np.mean(gram)))
    #
    #
    results, predictions = evaluate_t5_on(sciq_dev_df, best_model,tokenizer, model_pp,tokenizer_pp, model_name=model,
                                                                                        n_samples=n_samples)

    df = pd.DataFrame.from_dict(results)
    df.to_csv("results_sciq"+str(model)+"FINETUNED_ON_SQUAD9.csv", encoding='utf-8', index=False)

    f1_mean = results['f1']
    bleu_4 = results['bleu_4']
    bleu_1 = results['bleu_1']
    bleu_2 = results['bleu_2']
    bleu_3 = results['bleu_3']
    ppl_scores = results['ppl_scores']
    divs = results['divs']
    grammer = results['grammer']
    stat = sum(results['stats']) / len(results['stats'])
    words = [x[0] for x in results['words'] if len(x)]
    count = sum(results['count'])

    print(f'Mean scores {data_3}: {np.mean(f1_mean)}')
    print(f'BLEU 4 {data_3}: {np.mean(bleu_4)}')
    # print(f'ROUGE score {data_3}: {np.mean(rouge_scores, axis=0)}')
    print("Bleu 1: " + str(np.mean(bleu_1)))
    print("Bleu 2: " + str(np.mean(bleu_2)))
    print("Bleu 3: " + str(np.mean(bleu_3)))
    print(f'Mean PPL scores {data_3}: {np.mean(ppl_scores)}')
    print("Diversity " + str(np.mean(divs)))
    print("Grammer Error " + str(np.mean(grammer)))
    print("stat " + str(stat))
    print("words " + str(words))
    print("sum " + str(count))
    #
    results, predictions = evaluate_t5_on(squa_dev_df, best_model, tokenizer, model_pp=model_pp, tokenizer_pp=tokenizer_pp, model_name=model, n_samples=n_samples)

    df = pd.DataFrame.from_dict(results)
    df.to_csv("results_squad"+str(model)+"FINETUNED_ON_SQUAD9.csv", encoding='utf-8', index=False)

    f1_mean = results['f1']
    bleu_4 = results['bleu_4']
    bleu_1 = results['bleu_1']
    bleu_2 = results['bleu_2']
    bleu_3 = results['bleu_3']
    ppl_scores = results['ppl_scores']
    divs = results['divs']
    grammer = results['grammer']
    stat = sum(results['stats'])/len(results['stats'])
    words = [x for x in results['words'] if len(x)]
    words = list(np.array(words).flatten())
    count = sum(results['count'])

    print("Model " + str(model))
    print(f'Mean scores {data_1}: {np.mean(f1_mean)}')
    print(f'BLEU 4 {data_1}: {np.mean(bleu_4)}')
    print("Bleu 1: " + str(np.mean(bleu_1)))
    print("Bleu 2: " + str(np.mean(bleu_2)))
    print("Bleu 3: " + str(np.mean(bleu_3)))
    print(f'Mean PPL scores {data_1}: {np.mean(ppl_scores)}')
    print("Diversity " + str(np.mean(divs)))
    # print(f'ROUGE score {data_1}: {np.mean(rouge_scores, axis=0)}')
    print("Grammer Error " + str(np.mean(grammer)))
    print("stat " + str(stat))
    print("words " + str(words))
    print("sum " + str(count))


    results, predictions = evaluate_t5_on(race_dev_df, best_model, tokenizer, model_pp=model_pp, tokenizer_pp=tokenizer_pp, model_name=model, n_samples=n_samples)

    df = pd.DataFrame.from_dict(results)
    df.to_csv("results_race" + str(model) + "FINETUNED_ON_SQUAD9.csv", encoding='utf-8', index=False)

    f1_mean = results['f1']
    bleu_4 = results['bleu_4']
    bleu_1 = results['bleu_1']
    bleu_2 = results['bleu_2']
    bleu_3 = results['bleu_3']
    ppl_scores = results['ppl_scores']
    divs = results['divs']
    grammer = results['grammer']
    stat = sum(results['stats']) / len(results['stats'])
    words = [x for x in results['words'] if len(x)]
    words = list(np.array(words).flatten())
    count = sum(results['count'])

    print(f'Mean scores {data_2}: {np.mean(f1_mean)}')
    print(f'BLEU 4 {data_2}: {np.mean(bleu_4)}')
    # print(f'ROUGE score {data_2}: {np.mean(rouge_scores, axis=0)}')
    print("Bleu 1: "+ str(np.mean(bleu_1)))
    print("Bleu 2: " + str(np.mean(bleu_2)))
    print("Bleu 3: " + str(np.mean(bleu_3)))
    print(f'Mean PPL scores {data_2}: {np.mean(ppl_scores)}')
    print("Diversity " + str(np.mean(divs)))
    print("Grammer Error " + str(np.mean(grammer)))
    print("stat " + str(stat))
    print("words " + str(words))
    print("sum " + str(count))




    if debug:
        # print("SQUAD-DEBUG")
        # inspect(squa_dev_df, best_model, tokenizer, indices=[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15], model_name=model)
        # print("#########################################")
        # print("RACE-DEBUG")
        # inspect(race_dev_df, best_model, tokenizer, indices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], model_name=model)
        print("#########################################")
        print("SCIQ-DEBUG")
        inspect(sciq_dev_df, best_model, tokenizer, indices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], model_name=model)
        print("#########################################")


