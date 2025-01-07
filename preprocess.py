import re
from pathlib import Path
import nltk
from nltk.stem import PorterStemmer

text_file = Path("Data/Dataset/TextProcessing/61085-0.txt")


def preprocess_text():
    with open(text_file) as file:
        text = file.read()

    text = text.lower()

    text = remove_special_chars(text, get_text_special_chars(text))

    save_words(text)
    tokens = save_types_with_count(text)
    stemming(tokens)


def save_words(text):
    words = re.split(r"\s+", text)
    save_word_array(words, 'words.txt')


# def save_tokens(text, filename):
#     words = text.split(' ')
#     tokens = list(set(words))
#
#     text_counts = []
#     for token in tokens:
#         text_counts.append(f'{token}: {text.count(token)}')
#
#     with open(filename, "w") as file:
#         for token in tokens:
#             file.write(token + " ")
#
#     return tokens


def save_types_with_count(text):
    words = re.split(r"\s+", text)
    types_list = list(set(words))
    tokens = list(words)

    text_counts = []
    for type_item in types_list:
        text_counts.append(f'{type_item}: {text.count(type_item)}')

    save_word_array(tokens, 'tokens.txt')
    save_word_array(text_counts, 'types_count.txt')
    return tokens


def stemming(tokens):
    stemmer = PorterStemmer()
    stemmed_words = []
    for token in tokens:
        stemmed_words.append(stemmer.stem(token))

    save_word_array(stemmed_words, 'stemmed_words.txt')


def save_word_array(word_array, filename):
    saving_file = open(filename, 'w')
    for item in word_array:
        saving_file.write(item)
        saving_file.write('\n')
    saving_file.close()

    print(f'Data saved as {filename}')


def get_text_special_chars(text):
    special_chars_regex = r'[^a-zA-Z0-9 ]'
    special_chars = re.findall(special_chars_regex, text)
    special_chars = list(set(special_chars))
    return special_chars


def remove_special_chars(text, special_chars):
    for i in special_chars:
        text = text.replace(i, ' ')
    return text
