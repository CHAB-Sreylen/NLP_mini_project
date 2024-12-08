import random
import nltk
from nltk import FreqDist, ngrams, word_tokenize
from sklearn.model_selection import train_test_split
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')

# **1. Load Custom Dataset**
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# **2. Preprocess Corpus: Tokenize and Limit Vocabulary**
def preprocess_corpus(corpus, vocab_size):
    tokenized_corpus = word_tokenize(corpus)  # Tokenize using NLTK
    token_freq = Counter(tokenized_corpus)  # Calculate token frequencies
    vocab = set([token for token, _ in token_freq.most_common(vocab_size)])  # Limit vocabulary
    processed_corpus = [token if token in vocab else "<UNK>" for token in tokenized_corpus]  # Replace rare words
    return processed_corpus, vocab

# **3. Split the Corpus**
def split_corpus(corpus, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    train_data, test_data = train_test_split(corpus, test_size=test_ratio, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_ratio / (1 - test_ratio), random_state=42)
    print(f"Train_data: {len(train_data)}")
    print(f"val_data: {len(val_data)}")
    print(f"test_data: {len(test_data)}")
    return train_data, val_data, test_data

# **4. Train LM1: Backoff Model**
def train_backoff_lm(training_data, n=3):
    ngram_counts = FreqDist(ngrams(training_data, n))
    lower_ngram_counts = FreqDist(ngrams(training_data, n-1))

    def backoff_probability(ngram):
        if ngram in ngram_counts:
            return ngram_counts[ngram] / lower_ngram_counts[ngram[:-1]]
        elif len(ngram) > 1:
            return backoff_probability(ngram[1:])
        else:
            return 0  # OOV handling

    return backoff_probability

# **5. Train LM2: Interpolation with Add-k Smoothing**
def train_interpolation_lm(training_data, n=4, lambdas=None, k=0.1):
    vocab = set(training_data)
    V = len(vocab)
    ngram_counts = [FreqDist(ngrams(training_data, i)) for i in range(1, n+1)]
    lambdas = lambdas or [1/n] * n  # Default: uniform weights

    def smoothed_probability(ngram):
        prob = 0
        for i in range(len(ngram)):
            count_context = ngram_counts[i-1][ngram[:-1]] if i > 0 else len(training_data)
            prob += lambdas[i] * (
                (ngram_counts[i][ngram] + k) /
                (count_context + k * V)
            )
        return prob

    return smoothed_probability

# **6. Text Generation**
def generate_text(model, initial_context, max_length=100, vocabulary=None):
    context = initial_context
    generated_text = list(context)
    print(f"---- Initial context: {context} ------")

    for _ in range(max_length - len(context)):
        candidates = [(word, model(context + (word,))) for word in vocabulary]
        candidates = [(word, prob) for word, prob in candidates if prob > 0]

        if not candidates:  # No viable next word
            break

        total_prob = sum(prob for _, prob in candidates)
        candidates = [(word, prob / total_prob) for word, prob in candidates]

        words, probs = zip(*candidates)
        next_word = random.choices(words, probs)[0]

        context = context[1:] + (next_word,) if len(context) == len(initial_context) else context + (next_word,)
        generated_text.append(next_word)

    return ' '.join(generated_text)

# **7. Main Execution**
if __name__ == "__main__":
    # Replace 'your_dataset.txt' with the path to your dataset file
    file_path = 'data/cleaned_wiki_text.txt'
    
    # Load and preprocess dataset
    raw_corpus = load_dataset(file_path)
    vocab_size = 10000  # Adjust vocabulary size as needed
    processed_corpus, vocabulary = preprocess_corpus(raw_corpus, vocab_size)

    # print("Processed Corpus (first 50 tokens):", ' '.join(processed_corpus[:50]))
    print("Vocabulary Size:", len(vocabulary))
    print("raw_corpus Size:", len(raw_corpus))
    print("processed_corpus Size:", len(processed_corpus))

    # Split the corpus
    train_data, val_data, test_data = split_corpus(processed_corpus)

    n = 4  # n-gram size
    lm1 = train_backoff_lm(train_data, n)
    lm2 = train_interpolation_lm(train_data, n, lambdas=[0.1, 0.2, 0.3, 0.4], k=0.5)

    # Generate text
    initial_context = ("people in the","side")  # Example starting context
    max_length = 100

    print("\nGenerated Text:")
    print("LM1 (Backoff):", generate_text(lm1, initial_context, max_length, vocabulary))
    print("LM2 (Interpolation):", generate_text(lm2, initial_context, max_length, vocabulary))
