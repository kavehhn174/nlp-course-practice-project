import re
from os import listdir
from pathlib import Path
import pathlib
import numpy as np


def get_all_files(path):
    file_data = []
    folders_list = listdir(path)
    for index, folder in enumerate(folders_list):
        file_data.append({'folder_name': folder})
        dataset_list = listdir(pathlib.PurePath(path, folder))
        dataset_files = []
        for dataset_item in dataset_list:
            if dataset_item.endswith('.txt'):
                dataset_files.append(dataset_item)
                # documents_count += 1
        file_data[index]['datasets'] = dataset_files

    return file_data


data_path = Path("Data/Dataset/Classification-Train And Test/")


def get_max_prob(data):
    prob_array = []
    for item in data:
        prob_array.append(item['prob'])

    arr = np.array(prob_array)

    return np.argmax(arr)


def get_document_counts(groups_data):
    document_count = 0
    for group in groups_data:
        document_count += len(group['datasets'])

    return document_count


def get_total_vocabs_count(groups_data, dataset_vocabs):
    total_vocabs_count = 0
    for group in groups_data:
        total_vocabs_count += len(dataset_vocabs[group['folder_name']])
    return total_vocabs_count


def get_all_vocabs(groups_data, dataset_vocabs):
    all_vocabs = []
    for group in groups_data:
        all_vocabs += dataset_vocabs[group['folder_name']]

    return list(set(all_vocabs))


def get_all_vocabs_probs(all_vocabs, datasets_vocabs_count, groups_data, dataset_vocabs, total_vocabs_count):
    all_vocabs_probs = {}
    for group in groups_data:
        class_words = list(dataset_vocabs[group['folder_name']])
        __temp_prob_dict = {'unknown_word': 1 / (group['tokens_count'] + total_vocabs_count)}
        for vocab in all_vocabs:
            if vocab in class_words:
                numerator = datasets_vocabs_count[group['folder_name']][vocab] + 1
                denominator = group['tokens_count'] + total_vocabs_count
                __temp_prob_dict[vocab] = numerator / denominator
            else:
                numerator = 1
                denominator = group['tokens_count'] + total_vocabs_count
                __temp_prob_dict[vocab] = numerator / denominator
            all_vocabs_probs[group['folder_name']] = __temp_prob_dict

    return all_vocabs_probs


def get_groups_prior(groups_data, document_count):
    for group in groups_data:
        group['document_prob'] = len(group['datasets']) / document_count

    return groups_data


def get_groups_merged_dataset(groups_data):
    datasets_text = {}
    for group in groups_data:
        temp_text = ''
        for dataset in group['datasets']:
            temp_text += open(pathlib.PurePath(data_path, group['folder_name'], dataset)).read()
            temp_text += ' '

        datasets_text[group['folder_name']] = temp_text

    return datasets_text


def get_dataset_vocabs(groups_data, datasets_text):
    dataset_vocabs = {}
    for group in groups_data:
        tokens = datasets_text[group['folder_name']].split(' ')
        dataset_vocabs[group['folder_name']] = list(set(tokens))

    return dataset_vocabs


def get_vocabs_count(groups_data, datasets_text):
    datasets_vocabs_count = {}
    for group in groups_data:
        tokens = datasets_text[group['folder_name']].split(' ')
        group['tokens_count'] = len(tokens)
        vocabs = set(list(tokens))
        __temp_vocabs_dict = {}
        for vocab in vocabs:
            vocab_count = 0
            for token in tokens:
                if token == vocab:
                    vocab_count += 1
            __temp_vocabs_dict[vocab] = vocab_count

        datasets_vocabs_count[group['folder_name']] = __temp_vocabs_dict

    return datasets_vocabs_count


def select_classification_text():
    groups_data = get_all_files(data_path)
    document_count = get_document_counts(groups_data)
    groups_data = get_groups_prior(groups_data, document_count)
    datasets_text = get_groups_merged_dataset(groups_data)
    datasets_vocabs_count = get_vocabs_count(groups_data, datasets_text)
    dataset_vocabs = get_dataset_vocabs(groups_data, datasets_text)
    # total_vocabs_count = get_total_vocabs_count(groups_data, dataset_vocabs)
    all_vocabs = get_all_vocabs(groups_data, dataset_vocabs)
    total_vocabs_count = len(all_vocabs)
    all_vocabs_probs = get_all_vocabs_probs(all_vocabs, datasets_vocabs_count, groups_data, dataset_vocabs,
                                            total_vocabs_count)

    print('\n')
    print("Select a mode:")
    print("1. Custom Input Text")
    print("2. Load Test Files")
    selected_mode = input("Enter your selection (1, or 2): ")

    text = ''
    test_files = {}
    classified_files = {}
    test_files_count = 0
    folders_list = listdir(data_path)
    if selected_mode == "1":
        text = input('Enter Your Text \n')
        found_class = classify(text, groups_data, datasets_vocabs_count, dataset_vocabs, total_vocabs_count,
                               all_vocabs_probs)
        print('Your Text Classified As: ', found_class)

    elif selected_mode == "2":
        for folder in folders_list:
            __temp_test_array = listdir(pathlib.PurePath(data_path, folder + '/test'))
            test_files_count += len(__temp_test_array)
            test_files[folder] = __temp_test_array

        print(test_files)

        for folder in folders_list:
            __temp_classified_array = []
            for file in test_files[folder]:
                test_path = pathlib.PurePath(data_path, folder + '/test', file)
                with open(test_path) as f:
                    text = f.read()
                    __temp_classified_array.append(
                        classify(text, groups_data, datasets_vocabs_count, dataset_vocabs, total_vocabs_count,
                                 all_vocabs_probs))
            classified_files[folder] = __temp_classified_array

        accuracy = calculate_accuracy(classified_files, test_files_count, folders_list)

        print(classified_files)
        print('Accuracy:', accuracy, '%')


def calculate_accuracy(classified_files, test_files_count, folders_list):
    correct_predictions = 0
    for folder in folders_list:
        for item in classified_files[folder]:
            if item == folder:
                correct_predictions += 1

    return (correct_predictions / test_files_count) * 100


def classify(text, groups_data, datasets_vocabs_count, dataset_vocabs, total_vocabs_count, all_vocabs_probs):
    words = re.split(' ', text)
    words_probs_dict = {}
    words_class_data = []
    for __group in groups_data:
        __temp_prob_array = []
        for word in words:
            if word in all_vocabs_probs[__group['folder_name']]:
                __temp_prob_array.append(all_vocabs_probs[__group['folder_name']][word])
            else:
                __temp_prob_array.append(all_vocabs_probs[__group['folder_name']]['unknown_word'])
        words_probs_dict[__group['folder_name']] = __temp_prob_array

    for group in groups_data:
        __temp_prob = np.log10(group['document_prob'])
        for item in words_probs_dict[group['folder_name']]:
            __temp_prob += np.log10(item)

        group['prob'] = __temp_prob

    max_index = get_max_prob(groups_data)

    return groups_data[max_index]['folder_name']
