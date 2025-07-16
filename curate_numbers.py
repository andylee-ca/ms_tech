"""
Functions for detecting different types of numbers in text, including Roman numerals.
"""

import re

import nltk

# Constants and Patterns

# Regex for matching roman numerals
ROMAN_NUMERAL_PATTERN = re.compile(
    r"^(M{0,3})(CM|CD|D?C{0,3})" r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.IGNORECASE
)


# Functions for Curating Numbers


def is_valid_roman(string):
    """
    Checks if the input text is a valid Roman numeral.

    Args:
        text (str): The input string to check.

    Returns:
        bool: True if the text is a valid Roman numeral, False otherwise.
    """
    return bool(ROMAN_NUMERAL_PATTERN.match(string))


def is_noun_roman_bigram(bigram):
    """
    Checks if the bigram is (noun, valid Roman numeral).

    Args:
        bigram (tuple): Two words.

    Returns:
        bool: True if first is noun and second is valid Roman numeral.
    """
    return is_noun(bigram[0]) and is_valid_roman(bigram[1])


def find_valid_roman_numerals(input_text, debug=False):
    """
    Parses the input text to find and return valid Roman numerals.

    Args:
        input_text (str): The input string to parse.

    Returns:
        list: A list of valid Roman numerals found in the input text.
    """
    # Split the input text into words
    # input_tokens = nltk.word_tokenize(input_text)
    input_tokens = input_text.split()
    valid_roman_numerals = []

    for idx, token in enumerate(input_tokens):
        # Check if the word is a valid Roman numeral
        if debug:
            print(f"Checking token: {token} at index {idx}")

        # If entire token is not uppercase, skip it
        if not token.isupper():
            continue

        # Second validation for short Roman numerals particularly "I", "V", "X"
        if len(token) <= 2:

            # a valid Roman numeral on first position is unlikely to be a valid Roman numeral
            if idx == 0:
                continue

            # Extract first_word as the previous token before the roman numeral
            # Second word is the current token; the roman numeral itself
            previous_token, current_token = input_tokens[idx - 1], token
            
            # If previous token ends in comma or period, skip it
            if previous_token.endswith((",", ".", ";", ":")):
                if debug:
                    print(f"Skipping due to punctuation: {previous_token}")
                continue

            # Remove any symbols from previous token, as that may trigger false positives
            previous_token = re.sub(r"[^\w\s]", "", previous_token)

            # Check if the paired tokens are NP + Roman numeral            
            if is_noun_roman_bigram((previous_token, current_token)):
                # If the previous word is a noun and the current word is a valid Roman numeral
                if debug:
                    print(
                        f"Found valid Roman numeral: {previous_token} {current_token} "
                        f"-- adding {token}"
                    )

                # Append the current token as a likely valid Roman numeral
                valid_roman_numerals.append(token)

        else:
            if is_valid_roman(token):
                valid_roman_numerals.append(token)

    if debug:
        print(f"Original input: {input_text}")
        print(f"Valid Roman numerals found: {valid_roman_numerals}")

    # Filter and return only valid Roman numerals
    return valid_roman_numerals


def is_noun(word):
    """
    Determine if a given word is a noun using NLTK's part-of-speech tagging.
    Args:
        word (str): The word to check.
    Returns:
        bool: True if the word is tagged as a noun, False otherwise.
    Note:
        This function uses the Penn Treebank POS tagset, where noun tags start with 'NN'.
    """
    pos = nltk.pos_tag([word])[0][1]
    # Nouns in Penn Treebank tagset start with 'NN'
    return pos.startswith("NN")
