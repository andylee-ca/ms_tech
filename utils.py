import re
import pandas as pd

NUMBER_OF_SAMPLES_TO_SHOW = 3


def load_json_data(file_path):
    '''
    Loads JSON data from the specified file path into a pandas DataFrame.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded JSON data, or None if loading fails.
    '''
    try:
        data = pd.read_json(file_path, orient="records", lines=False)
        return data
    except ValueError as e:
        print(f"Error loading JSON data: {e}")
        return None


def strip_html_tags(text):
    """
    Strips HTML tags from a string.

    Args:
        text (str): The input string potentially containing HTML tags.

    Returns:
        str: The string with HTML tags removed.
    """
    # Regex to find any HTML tag: < followed by any characters, then >
    # The '?' makes it non-greedy, matching the shortest possible string.
    # The '|' handles self-closing tags like <br/> or <img src="...">
    # and also comments (though less common in user-generated text usually)
    clean = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(clean, '', text)


def display_sample_questions(df, lookup_column = None, extra_details_col = None, sample_count = 3):
    """
    Displays a sample of questions from the DataFrame where the specified lookup_column is True.

    Args:
        df (pd.DataFrame): The DataFrame containing the questions.
        lookup_column (str, optional): The column name to use as a boolean filter for selecting rows.
        sample_count (int, optional): The number of sample questions to display. Default is 3.

    Raises:
        ValueError: If lookup_column is None or empty.

    Prints:
        The index, category, and question text for each sampled row.
    """
    if lookup_column is None:
        raise ValueError("lookup_column argument cannot be empty or None")
    
    if df is None or lookup_column not in df.columns:
        raise ValueError(f"DataFrame is None or lookup_column '{lookup_column}' does not exist in DataFrame")
    
    if extra_details_col is not None and extra_details_col not in df.columns:
        raise ValueError(f"show_supplementary_column '{extra_details_col}' does not exist in DataFrame")
    
    if extra_details_col is None:
        print(f"Sampling questions with '{lookup_column}' set to True:")

        sample_list = df[df[lookup_column]].sample(sample_count)[['category', 'question']]
        for idx, row in sample_list.iterrows():
            print(f"{idx} - [{row['category']}] {row['question']}")

    else:
        print(f"Sampling questions with '{lookup_column}' set to True, showing '{extra_details_col}':")

        sample_list = df[df[lookup_column]].sample(sample_count)[['category', 'question', extra_details_col]]

        for idx, row in sample_list.iterrows():
            print(f"{idx} - [{row['category']}] {row['question']} (Extra: {row[extra_details_col]})")

    print()


def strip_quotes(text):
    """
    Strips quotes only if it appears at the start and end of the argument text.
    """
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text