import pandas as pd
import re

# import os
# os.chdir('/content/drive/MyDrive/Language Technology Project/Data')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# device = args.gpu

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

arguments_test = pd.read_csv("arguments-test.tsv", sep="\t")
arguments_training = pd.read_csv("arguments-training.tsv", sep="\t")
arguments_validation = pd.read_csv("arguments-validation.tsv", sep="\t")

"""# Evaluation: diversity (unique n-grams per generated tokens), fidelity (BLEU)"""

from nltk.util import ngrams
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def tokenize_sentences(sentences):
  tokenized = []
  if isinstance(sentences, list):
    for sentence in sentences:
      tokenized.append(sentence.split(" "))
  else:
    tokenized = sentences.split(" ")
  return tokenized

def get_unique_ngrams(paraphrases, n):
  tokenized = tokenize_sentences(paraphrases)

  sentence_ngrams = []
  for sentence in tokenized:
    sentence_ngrams.append(list(ngrams(sentence, n)))

  sentence_ngrams = [item for sublist in sentence_ngrams for item in sublist]

  all_ngrams = len(sentence_ngrams)
  unique_ngrams = len(set(sentence_ngrams))

  return unique_ngrams

"""# Integrate text generation and evaluation, add to dataframe"""

from tqdm import tqdm
from statistics import mean
from nltk.translate.bleu_score import SmoothingFunction

def paraphrase_and_evaluate(args):
  df = pd.read_csv(args.csv_path, sep='\t')

  df_with_scores = pd.DataFrame()
  smooth = SmoothingFunction()

  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Generate paraphrases
    paraphrases = paraphrase(row["Premise"],
    args.num_beams,
    args.num_beam_groups,
    args.num_return_sequences,
    args.repetition_penalty,
    args.diversity_penalty,
    args.no_repeat_ngram_size,
    args.temperature,
    args.max_length)

    # Clean paraphrases (eval methods are case sensitive)
    paraphrases_clean = [phrase.lower() for phrase in paraphrases]
    paraphrases_clean = [re.sub(r'[^\w\s]', '', phrase) for phrase in paraphrases]

    reference = re.sub(r'[^\w\s]', '', row["Premise"].lower())
    
    # Evaluate
    unique_ngrams = []
    for n in range(1,5):
      # print(n)
      unique_ngrams.append(get_unique_ngrams(paraphrases_clean, n))

    bleu_scores = []
    for phrase in paraphrases_clean:
      bleu_scores.append(sentence_bleu([reference], phrase, smoothing_function=smooth.method1))

    avg_unique_ngrams = mean(unique_ngrams)
    avg_bleu_score = mean(bleu_scores)

    # Add paraphrases and bleu scores to df
    for phrase_index in range(len(paraphrases)):
      df_with_scores.loc[index, f'paraphrase_{phrase_index}'] = paraphrases[phrase_index]
      df_with_scores.loc[index, f'bleu_score_{phrase_index}'] = bleu_scores[phrase_index]

    for ngram_index in range(len(unique_ngrams)):
      df_with_scores.loc[index, f'ngram_{ngram_index+1}'] = unique_ngrams[ngram_index]

    df_with_scores.loc[index, 'avg_unique_ngrams'] = avg_unique_ngrams
    df_with_scores.loc[index, 'avg_bleu_score'] = avg_bleu_score

    df_with_scores.loc[index, 'original'] = row["Premise"]
    

    # print(paraphrases, unique_ngrams, bleu_scores, avg_unique_ngrams, avg_bleu_score)

  print(df_with_scores.mean())
  return df_with_scores

i = arguments_training.sample(n=100, random_state=1)

test_results = paraphrase_and_evaluate(test_arguments,num_beams=10,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128)
