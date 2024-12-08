import random
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
import math

# Step 1: Load and Split the Text Corpus
with open("cleaned_wiki_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split into lines, shuffle, and divide into train, validation, and test sets
lines = text.split("\n")
random.shuffle(lines)

train_size = int(0.7 * len(lines))
val_size = int(0.1 * len(lines))

train_set = lines[:train_size]
val_set = lines[train_size:train_size + val_size]
test_set = lines[train_size + val_size:]

train_text = " ".join(train_set)
val_text = " ".join(val_set)
test_text = " ".join(test_set)

# Step 2: Tokenize and Replace Out-of-Vocabulary Tokens
def replace_oov(tokens, vocab):
    return [token if token in vocab else "<UNK>" for token in tokens]

# Tokenize and limit vocabulary size
def preprocess_and_tokenize(text, vocab_size=5000):
    tokens = word_tokenize(text.lower())
    word_counts = Counter(tokens)
    vocab = {word for word, _ in word_counts.most_common(vocab_size)}
    return replace_oov(tokens, vocab), vocab

train_tokens, vocab = preprocess_and_tokenize(train_text)
val_tokens = replace_oov(word_tokenize(val_text.lower()), vocab)
test_tokens = replace_oov(word_tokenize(test_text.lower()), vocab)

# Step 3: Build N-gram Models
def build_ngram_model(tokens, n):
    ngrams_list = list(ngrams(tokens, n))
    return FreqDist(ngrams_list)

lm1_4gram = build_ngram_model(train_tokens, 4)
models = {i: build_ngram_model(train_tokens, i) for i in range(1, 5)}

# Step 4: Backoff Probability Function
def backoff_prob(model, ngram):
    if ngram in model:
        return model[ngram] / sum(model.values())
    else:
        return 1e-10  # Small probability for unseen n-grams

# Step 5: Interpolation with Add-k Smoothing
def interpolate_prob(ngram, models, lambdas, k):
    n = len(ngram)
    prob = 0
    for i in range(1, n + 1):
        prefix_ngram = ngram[-i:]
        count = models[i].get(prefix_ngram, 0) + k
        total = sum(models[i].values()) + k * len(models[i])
        prob += lambdas[i - 1] * (count / total)
    return prob

# Hyperparameters for interpolation
lambdas = [0.1, 0.2, 0.3, 0.4]
k = 0.5

# Step 6: Perplexity Calculation
def perplexity(test_tokens, model, prob_func):
    n = len(next(iter(model)))  # N-gram size
    test_ngrams = list(ngrams(test_tokens, n))
    log_prob_sum = 0
    for ngram in test_ngrams:
        prob = prob_func(model, ngram)
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / len(test_ngrams))

# Evaluate perplexity for LM1 and LM2
lm1_perplexity = perplexity(test_tokens, lm1_4gram, backoff_prob)
lm2_perplexity = perplexity(test_tokens, models, lambda m, ng: interpolate_prob(ng, models, lambdas, k))

print(f"LM1 Perplexity (Backoff): {lm1_perplexity}")
print(f"LM2 Perplexity (Interpolation): {lm2_perplexity}")

# Step 7: Text Generation
def generate_text_backoff(model, start_token, length=20):
    n = len(next(iter(model)))
    current_tokens = start_token.split()[-(n - 1):]
    generated_text = current_tokens[:]
    
    for _ in range(length):
        candidates = [ngram for ngram in model if ngram[:-1] == tuple(current_tokens)]
        if candidates:
            next_word = random.choices(candidates, weights=[model[c] for c in candidates])[0][-1]
        else:
            next_word = "<UNK>"
        generated_text.append(next_word)
        current_tokens = generated_text[-(n - 1):]
    
    return " ".join(generated_text)

def generate_text_interpolation(models, start_token, length=20):
    n = max(models.keys())
    current_tokens = start_token.split()[-(n - 1):]
    generated_text = current_tokens[:]
    
    for _ in range(length):
        next_word_probs = {}
        for word in vocab:
            next_word = tuple(current_tokens + [word])
            next_word_probs[word] = interpolate_prob(next_word, models, lambdas, k)
        
        next_word = max(next_word_probs, key=next_word_probs.get)
        generated_text.append(next_word)
        current_tokens = generated_text[-(n - 1):]
    
    return " ".join(generated_text)

# Example: Generate text using both models
start_token = "The history of"
print("\nGenerated Text (LM1 - Backoff):")
print(generate_text_backoff(lm1_4gram, start_token))

print("\nGenerated Text (LM2 - Interpolation):")
print(generate_text_interpolation(models, start_token))
