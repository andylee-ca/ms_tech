"""
Main script for curating and exporting Jeopardy question features.
"""
import re
import os
import nltk

from curate_numbers import find_valid_roman_numerals
from curate_non_english import find_non_english_word, lemmatizer
from curate_unusual_proper_nouns import find_proper_nouns, generate_word_frequency
from utils import load_json_data, strip_html_tags, strip_quotes, display_sample_questions

# Threshold for low frequency words, used in filtering unusual proper nouns
LOW_FREQUENCY_THRESHOLD = 2

# Spelled-out numbers and their variations
SPELLED_NUMBERS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion"
]

SPELLED_NUMBERS += [
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth',
    'ninth', 'tenth'
]
SPELLED_NUMBERS += [
    'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth',
    'seventeenth', 'eighteenth', 'nineteenth'
]
SPELLED_NUMBERS += [
    'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth', 'seventieth',
    'eightieth', 'ninetieth'
]
SPELLED_NUMBERS += [
    'hundredth', 'thousandth', 'millionth', 'billionth'
]
SPELLED_NUMBERS += [
    'twice', 'thrice', 'once'
]
SPELLED_NUMBERS += [
    'single', 'double', 'triple', 'quadruple', 'quintuple', 'sextuple',
    'septuple', 'octuple', 'nonuple', 'decuple'
]
SPELLED_NUMBERS += [
    'dozen', 'fortnight', 'score', 'century', 'millennium'
]

# Set of spelled-out numbers for quick lookup
SPELLED_NUMBERS_SET = set(SPELLED_NUMBERS)

# Path to the JSON file containing Jeopardy questions
JSON_FILE_PATH = './dataset/JEOPARDY_QUESTIONS1.json'


def main():
    # Load your data
    df = load_json_data(JSON_FILE_PATH)

    # Store list of feature names from the fresh import for use later
    original_columns = df.columns.tolist()

    # Pre-cleaning steps

    # duplicate 'question' to 'original_question' to preserve original text
    df['original_question'] = df['question'].copy()

    # Strip HTML tags from the 'question' column
    df['question'] = df['question'].apply(strip_html_tags)

    # Strip leading and trailing quotes from the 'question' column
    df['question'] = df['question'].apply(strip_quotes)


    # --- Curate Numbers ---
    # Check for spelled numbers in the 'question' column
    df['has_spelled_number'] = df['question'].apply(
        lambda x: bool(set(nltk.word_tokenize(x.lower())) & SPELLED_NUMBERS_SET)
    )

    # Extract roman numerals in the 'question' column
    df['roman_text'] = df['question'].apply(
        lambda x: find_valid_roman_numerals(x)
    )
    df['has_roman_numeral'] = df['roman_text'].apply(
        lambda x: len(x) > 0
    )

    # Check for numerical values in the 'question' column
    df['has_numerical_value'] = df['question'].apply(
        lambda x: any(re.search(r'\d', token) for token in nltk.word_tokenize(x))
    )

    # Combine all number-related flags into a single column
    df['has_number'] = df['has_spelled_number'] | df['has_numerical_value'] | df['has_roman_numeral']

    # Print curated number samples
    display_sample_questions(df, 'has_spelled_number')
    display_sample_questions(df, 'has_roman_numeral')
    display_sample_questions(df, 'has_numerical_value')


    # --- Curate Non-English Words ---
    print("Parsing questions for non-English words (this may take few minutes)...", end=' ')

    df['non_english_words'] = df['question'].apply(
        lambda x: find_non_english_word(x, method="combined", lemmatizer=lemmatizer, debug=False)
    )

    print("Parsing finished.", end='\n')

    df['has_non_english_word'] = df['non_english_words'].apply(
        lambda x: len(x) > 0
    )

    # Print non-English word samples
    display_sample_questions(df, lookup_column='has_non_english_word', extra_details_col='non_english_words')


    # --- Curate Unusual Proper Nouns ---
    print("Parsing questions for unusual proper nouns (this may take few minutes)...", end=' ')

    df['proper_nouns'] = df['question'].apply(
        lambda x: find_proper_nouns(x, debug=False)
    )

    # Generate word frequency DataFrame for the 'question' column
    frequency_df = generate_word_frequency(df, 'question')

    # Filter out unusual proper nouns based on frequency threshold
    unusual_words = set(frequency_df[frequency_df['frequency'] <= LOW_FREQUENCY_THRESHOLD]['word'].tolist())

    # Check if proper nouns are unusual using set intersection
    df['has_unusual_proper_noun'] = df['proper_nouns'].apply(
        lambda x: bool(set(word.lower() for word in x) & unusual_words)
    )

    # Print unusual proper noun samples
    display_sample_questions(df, lookup_column='has_unusual_proper_noun', extra_details_col='proper_nouns', sample_count=10)


    # --- Export Curated Samples ---

    # Sample 1000 based on the 'has_number' column, using the original columns
    df_with_numbers = df[df['has_number']].sample(n=1000, random_state=42)[original_columns]

    # Check if the export directory exists, create it if not    
    if not os.path.exists('./export'):
        os.makedirs('./export')

    # Save the DF to JSON file
    df_with_numbers.to_json('./export/JEOPARDY_QUESTIONS_numbers.json', orient='records', lines=False)

    # Sample 1000 based on the 'has_non_english_word' column, using the original columns
    df_with_non_english = df[df['has_non_english_word']].sample(n=1000, random_state=42)[original_columns]

    # Save the DF to JSON file
    df_with_non_english.to_json('./export/JEOPARDY_QUESTIONS_non_english.json', orient='records', lines=False)

    # Sample 1000 based on the 'has_unusual_proper_noun' column, using the original columns
    df_with_unusual_proper_nouns = df[df['has_unusual_proper_noun']].sample(n=1000, random_state=42)[original_columns]

    # Save the DF to JSON file
    df_with_unusual_proper_nouns.to_json('./export/JEOPARDY_QUESTIONS_unusual_proper_nouns.json', orient='records', lines=False)


if __name__ == "__main__":
    main()
