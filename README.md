# Assessment - Andy Lee

## Project Overview
This project is designed to curate records from Jeopardy questions, focusing on identifying questions that match specific criteria: containing numbers, non-English words, or unusual proper nouns. The curation process involves multiple steps including data loading, preprocessing, criteria-based filtering, and exporting curated samples.

## How to Run

To execute the curation workflow, simply run:

```
python main.py
```

This process will take a few minutes and will generate curated JSON files in the `export/` directory.

## Curation Process Overview

The curation process consists of several key steps to filter and export records (questions) that meet defined criteria:

1. **Data Loading & Preprocessing:**  
   The script loads questions from a JSON file, strips HTML tags, and removes leading/trailing quotes.

2. **Number Feature Curation:**  
   There are three sub-tasks in this step. It detects the following types of numbers in the questions:
   - spelled-out numbers
   - Roman numerals
   - numerical values

   Then, the three flags are combined into a single `has_number` column.

3. **Non-English Word Detection:**  
   Using NLTK, the script first tags each word in the questions. It then identifies non-English words by checking against WordNet then the NLTK English vocabulary, excluding common stopwords and certain parts of speech. This helps in filtering out words that are likely not part of the English language.
   
   The reason both WordNet and NLTK's vocabulary are used is to ensure a comprehensive check against both common and less common words, providing a more robust detection mechanism.

4. **Unusual Proper Noun Extraction:**  
   For this project, "unusual" are defined as words that have appeared 2 or fewer times across all questions.

   First, the script concatenates all questions into a single string, tokenizes it into words, and calculates the frequency of each word. Any words appearing two or fewer times are stored into a set.

   Then, each question is tagged and proper nouns are extracted. If a proper noun is in the set of unusual words, it is considered an unusual proper noun.

5. **Exporting Curated Samples:**  
   For each criterion, 1000 matching questions are sampled and exported to individual JSON files in the `export/` directory.

## Dependencies
All required Python packages are listed in `requirements.txt`.

To install, run:

```
pip install -r requirements.txt
```

## Modules Used
This project uses the following modules and libraries:

- `nltk`: Natural language processing (tokenization, POS tagging, etc.)
- `re`: Regular expression for text pattern matching
- `os`: File and directory operations
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations
- `curate_numbers`: Functions for identifying numbers and Roman numerals in text
- `curate_non_english`: Functions for detecting non-English words using NLTK
- `curate_unusual_proper_nouns`: Functions for extracting and analyzing proper nouns
- `utils`: Utility functions for loading JSON data, cleaning text, and displaying sample questions


## File Structure
- `main.py`: The main script that orchestrates the curation process.
- `curate_non_english.py`: Contains functions for detecting non-English words.
- `curate_numbers.py`: Contains functions for identifying numbers in text.
- `curate_unusual_proper_nouns.py`: Contains functions for extracting unusual proper nouns.
- `utils.py`: Contains miscellaneous functions for loading and cleaning data.
- `workbook.ipynb`: Jupyter notebook for exploratory analysis and prototyping.
- `dataset/`: Contains the input JSON file of Jeopardy questions.
- `export/`: Output directory for curated JSON files.

## Output
Curated samples are exported to:
- `export/JEOPARDY_QUESTIONS_numbers.json`
- `export/JEOPARDY_QUESTIONS_non_english.json`
- `export/JEOPARDY_QUESTIONS_unusual_proper_nouns.json`

Each file contains a sample of 1000 questions where the "question" field matches the respective criterion.