import re
from pathlib import Path
import enchant
import nltk
from strsimpy.damerau import Damerau
import languagemodel
from nltk.util import pad_sequence, bigrams
import json
import pickle
import ast

d = enchant.Dict("en_US")

text_file = Path("Data/Dataset/Spelling Dataset/test/spell-testset.txt")
dictionary_file = Path("Data/Dataset/Spelling Dataset/test/Dictionary/dictionary.data")
candidate_file = Path("Data/Dataset/Spelling Dataset/spell-errors.txt")
del_conf_mtx_file = Path("Data/Dataset/Spelling Dataset/test/Confusion Matrix/del-confusion.data")
ins_conf_mtx_file = Path("Data/Dataset/Spelling Dataset/test/Confusion Matrix/ins-confusion.data")
sub_conf_mtx_file = Path("Data/Dataset/Spelling Dataset/test/Confusion Matrix/sub-confusion.data")
trans_conf_mtx_file = Path("Data/Dataset/Spelling Dataset/test/Confusion Matrix/Transposition-confusion.data")
types_file = Path("spell_tokens.txt")
dataset_location = Path("Data/Dataset/Spelling Dataset/test/Dictionary/Dataset.data")


def get_tokens_from_file(file_path):
    with open(file_path) as f:
        tokens = f.read()
    return tokens


spell_types = get_tokens_from_file(types_file)

with open(dataset_location) as file:
    dataset = file.read()

with open(del_conf_mtx_file, "r") as file:
    deletion_string = file.read()
    deletion_dict = ast.literal_eval(deletion_string)

with open(ins_conf_mtx_file, "r") as file:
    insertion_string = file.read()
    insertion_dict = ast.literal_eval(insertion_string)

with open(sub_conf_mtx_file, "r") as file:
    sub_string = file.read()
    substitute_dict = ast.literal_eval(sub_string)

with open(trans_conf_mtx_file, "r") as file:
    transpos_string = file.read()
    transpos_dict = ast.literal_eval(transpos_string)


# typo_data = {}
# with open(candidate_file, "r") as f:
#     for line in f.readlines():
#         _word, wrongs = line.split(":")
#         typos = wrongs.replace(" ", "").split(",")
#         for typo in typos:
#             typo_data[typo.strip()] = _word


def spell_check():
    # with open(text_file) as file:
    #     text = file.read()
    text = input('Enter Your Sentence \n')

    words = re.split(' ', text)
    fixed_words = []
    for word in words:
        fixed_words.append(spell_check_by_word(word))

    print('Spell Corrected Text :', end='')
    for word in fixed_words:
        print(word + ' ', end='')


def spell_check_by_word(word):
    data = []
    word = word.lower()

    # Get A List Of Candidates -----------
    words_candidates = get_candidate(word)
    print(words_candidates)

    if is_real_word(word) and word.lower() not in words_candidates:
        words_candidates.append(word.lower())


    # print(words_candidates)
    for candidate in words_candidates:
        data.append({'word': word, 'candidate': candidate})
    # ------------------------------------

    # Get Unigram Probabilities -----------
    unigram_probs = []
    for index, candidate in enumerate(words_candidates):
        __temp_unigram = languagemodel.get_unigram_prob(candidate, spell_types, dataset)
        unigram_probs.append(__temp_unigram)
        data[index]['unigram_prob'] = __temp_unigram

    # print(unigram_probs)
    # -------------------------------------

    # Get Med Operator --------------------
    med_operators = []
    for index, candidate in enumerate(words_candidates):
        __temp_med_operator = get_med_operator(word, candidate)
        med_operators.append(__temp_med_operator)
        data[index]['med_operator'] = __temp_med_operator
    # -------------------------------------

    # Get The Characters For Searching Confusion Matrix -----------
    conf_matrix_chars = []
    for index, candidate in enumerate(words_candidates):
        __temp_conf_mtx_char = get_confusion_matrix_chars(med_operators[index], [word, candidate])
        conf_matrix_chars.append(__temp_conf_mtx_char)
        data[index]['conf_mtx_chars'] = __temp_conf_mtx_char
    # -------------------------------------------------------------

    # Get The Values For Those Characters In Confusion Matrix -----------
    conf_matrix_data = []
    for index, chars in enumerate(conf_matrix_chars):
        if chars != '##':
            __temp_conf_data = get_confusion_matrix_data(chars, med_operators[index])
            conf_matrix_data.append(__temp_conf_data)
            data[index]['conf_matrix_data'] = __temp_conf_data
        else:
            conf_matrix_data.append('##')
            data[index]['selection_prob'] = 0.95
    # --------------------------------------------------------------------

    # Calculate Selection Probability -----------
    selection_prob = []
    for index, item in enumerate(data):
        if item['conf_mtx_chars'] != '##':
            __temp_selection_data = get_selection_prob(item['conf_mtx_chars'], item['conf_matrix_data'], dataset)
            selection_prob.append(__temp_selection_data)
            data[index]['selection_prob'] = __temp_selection_data
        else:
            data[index]['selection_prob'] = 0.95
    # --------------------------------------------

    # Calculate Final Probability -----------
    final_probability = []
    for index, item in enumerate(data):
        __temp_final_prob = data[index]['unigram_prob'] * data[index]['selection_prob'] * 100000000000000
        final_probability.append(__temp_final_prob)
        data[index]['final_prob'] = __temp_final_prob

    # --------------------------------------------
    max_index = get_max_prob(data)

    return data[max_index]['candidate']


def get_max_prob(data):
    import numpy as np

    prob_array = []
    for item in data:
        prob_array.append(item['final_prob'])

    arr = np.array(prob_array)

    return np.argmax(arr)


def get_selection_prob(chars, chars_data, dataset):
    text_data = dataset.count(chars)
    return (chars_data + 1) / (text_data + len(spell_types))


def get_confusion_matrix_data(chars, operator):
    if operator == 'None':
        return 0
    my_dict = ''
    if operator == 'Deletion':
        my_dict = deletion_dict
    if operator == 'Insertion':
        my_dict = insertion_dict
    if operator == 'Transposition':
        my_dict = transpos_dict
    if operator == 'Substitute':
        my_dict = substitute_dict

    return my_dict[chars]


def get_confusion_matrix_chars(operator, two_words_array):
    first_word_array = list(two_words_array[0].lower())
    second_word_array = list(two_words_array[1].lower())
    if operator == 'Deletion':
        first_word_array.append('#')
        # print(first_word_array)
        # print(second_word_array)
        for index, s_letter in enumerate(second_word_array):
            if s_letter != first_word_array[index]:
                error_index = index
                if error_index == 0:
                    # print(second_word_array[error_index] + second_word_array[error_index + 1])
                    return second_word_array[error_index] + second_word_array[error_index + 1]
                else:
                    # print(second_word_array[error_index] + second_word_array[error_index - 1])
                    return second_word_array[error_index - 1] + second_word_array[error_index]

    if operator == 'Insertion':
        second_word_array.append('#')
        # print(first_word_array)
        # print(second_word_array)
        for index, f_letter in enumerate(first_word_array):
            if f_letter != second_word_array[index]:
                error_index = index
                if error_index == 0:
                    # print(first_word_array[error_index] + first_word_array[error_index + 1])
                    return first_word_array[error_index] + first_word_array[error_index + 1]
                else:
                    # print(first_word_array[error_index - 1] + first_word_array[error_index])
                    return first_word_array[error_index - 1] + first_word_array[error_index]
    if operator == 'Transposition':
        for index, f_letter in enumerate(first_word_array):
            if f_letter != second_word_array[index]:
                # print(first_word_array)
                # print(second_word_array)
                # print(first_word_array[index] + second_word_array[index])
                return second_word_array[index] + first_word_array[index]
    if operator == 'Substitute':
        for index, f_letter in enumerate(first_word_array):
            if f_letter != second_word_array[index]:
                # print(first_word_array)
                # print(second_word_array)
                # print(second_word_array[index] + first_word_array[index])
                return second_word_array[index] + first_word_array[index]

    return '##'


def get_med_operator(first_word, second_word):
    first_word = first_word.lower()
    second_word = second_word.lower()
    if len(first_word) == len(second_word):
        if first_word != second_word:
            different_chars = [i for i in range(len(first_word)) if first_word[i] != second_word[i]]
            if len(different_chars) == 2:
                if (first_word[different_chars[0]] == second_word[different_chars[1]] and
                        first_word[different_chars[1]] == second_word[different_chars[0]]):
                    return 'Transposition'
            else:
                return 'Substitute'
    else:
        if len(first_word) < len(second_word):
            return 'Deletion'
        else:
            return 'Insertion'

    return 'None'


def get_candidate(word):
    candidates = d.suggest(word.lower())
    candidates = [x.lower() for x in candidates]
    candidates = list(set(candidates))
    final_candidates = []
    print(word)

    print(candidates)

    damerau = Damerau()

    for candidate in candidates:
        if damerau.distance(candidate, word.lower()) <= 1 and candidate.isalpha():
            final_candidates.append(candidate.lower())

    if len(final_candidates) == 0:
        final_candidates.append(word.lower())

    return final_candidates


def is_real_word(word):
    return d.check(word)
    # with open(dictionary_file) as file:
    #     dictionary = file.read()
    #     dictionary_words = re.split(r'\W+', dictionary)
    #
    # if word in dictionary_words:
    #     return True
    # else:
    #     return False
