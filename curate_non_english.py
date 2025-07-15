import nltk

from nltk.corpus import wordnet, words, stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()

# Set of stopwords from NLTK
STOPWORDS = set(stopwords.words("english"))

# English vocabulary set for filtering unusual proper nouns
ENGLISH_VOCAB = set(w.lower() for w in words.words())


def find_non_english_word(
        input_text: str,
        method: str = "wordnet",
        stopword_list: set[str] | list[str] = STOPWORDS,
        lemmatizer: "WordNetLemmatizer" = lemmatizer,
        debug: bool = False
    ) -> list[str]:

    """
    Identify non-English words in the input text using POS tagging and lemmatization.

    Args:
        input_text (str): The input string to analyze.
        method (str, optional): Method to determine if a word is English.
            - "wordnet": Uses WordNet synsets (default).
            - "en_dict": Uses the NLTK English vocabulary word list.
        stopword_list (set or list of str): Required. A set or list of stopwords to ignore (e.g., set(stopwords.words("english"))).
        lemmatizer (WordNetLemmatizer): Required. An instance of WordNetLemmatizer must be provided by the caller.
        debug (bool, optional): If True, prints debug information. Default is False.

    Returns:
        list: A list of non-English words found in the input text.

    Notes:
        - Only alphabetic, non-stopword, and non-function-tagged words are checked.
        - POS tags excluded: NNP, NNPS, IN, DT, WP, WP$, WRB, PRP, PRP$, CC, TO, MD, EX, UH.
        - The method parameter controls the English word check:
            * "wordnet": A word is considered English if it has WordNet synsets.
            * "en_dict": A word is considered English if it is in the NLTK English vocabulary.
    """
    # Tags only used for tagged-based filtering
    excluded_pos_tags = ['NNP', 'NNPS', 'IN', 'DT', 'WP', 'WP$', 'WRB', 'PRP', 'PRP$', 'CC', 'TO', 'MD', 'EX', 'UH']

    # Tokenize and POS tag the full sentence
    input_tokens = nltk.word_tokenize(input_text)
    tagged_tokens = nltk.pos_tag(input_tokens)

    # Filter out tokens that are not alphabetic or are stopwords or in one of the predefined function tags
    filtered_tokens = [
        (word, tag) for word, tag in tagged_tokens
        if word.isalpha() and word.lower() not in stopword_list and tag not in excluded_pos_tags
    ]

    non_english_words = []
    for word, tag in filtered_tokens:
        lemma_word = lemmatizer.lemmatize(word.lower())
        if method == "wordnet" and not wordnet.synsets(lemma_word):
            # Check lemma words in WordNet only
            non_english_words.append(word)
        elif method == "en_dict" and not (lemma_word in ENGLISH_VOCAB or lemma_word.lower() in ENGLISH_VOCAB or lemma_word.upper() in ENGLISH_VOCAB or word.lower() in ENGLISH_VOCAB):
            # Check lemma words in the NLTK English vocabulary
            non_english_words.append(word)
        elif method == "combined":
            # Check both WordNet and NLTK English vocabulary
            in_wordnet = bool(wordnet.synsets(lemma_word))
            in_en_dict = lemma_word in ENGLISH_VOCAB or lemma_word.lower() in ENGLISH_VOCAB or lemma_word.upper() in ENGLISH_VOCAB or word.lower() in ENGLISH_VOCAB
            if not (in_wordnet or in_en_dict):
                non_english_words.append(word)

        if debug:
            print(f"Word: {word}, Tag: {tag}, Lemma: {lemma_word}, In WordNet: {bool(wordnet.synsets(lemma_word))}, In English Dict: {lemma_word in ENGLISH_VOCAB or lemma_word.lower() in ENGLISH_VOCAB or lemma_word.upper() in ENGLISH_VOCAB or word.lower() in ENGLISH_VOCAB}")

    if debug:
        print(f"Original input: {input_text}")
        print(f"Non-English words found: {non_english_words}")

    return non_english_words