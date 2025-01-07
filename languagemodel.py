from pathlib import Path
import preprocess
from nltk.util import pad_sequence, bigrams


def get_bigram_prob(two_word_array):
    dataset_location = Path("Data/Dataset/Spelling Dataset/test/Dictionary/Dataset.data")
    with open(dataset_location) as file:
        dataset = file.read()

    # print(two_word_array)

    first_word = two_word_array[0]
    second_word = two_word_array[1]

    first_count = dataset.lower().count(first_word)
    bigram_count = dataset.lower().count(first_word + ' ' + second_word)
    if bigram_count == 0:
        bigram_count = 1

    return bigram_count / first_count


def calculate_bigram(words, words_candidates):
    bigrams_list = list(pad_sequence(words_candidates, pad_left=False, pad_right=False, n=2))
    print(bigrams)

    words.insert(0, '<s>')
    words.append('</s>')

    bigram_list = []
    for index, word in enumerate(words):
        if word != '<s>' and word != '</s>':
            before_word = words[index - 1]
            mid_word = word
            after_word = words[index + 1]
            # print([before_word, mid_word, after_word])
            bigram_list.append(list(bigrams([before_word, mid_word, after_word])))
            # print(list(bigrams([before_word, mid_word, after_word])))

    for bigram_item in bigram_list:
        print(bigram_item)
        print(get_bigram_prob(bigram_item[0]) * get_bigram_prob(bigram_item[1]))


def get_unigram_prob(target_word, types, dataset):
    # dataset_location = Path("Data/Dataset/Spelling Dataset/test/Dictionary/Dataset.data")
    # with open(dataset_location) as file:
    #     dataset = file.read()

    all_words = dataset.lower().split(' ')
    word_count = 1
    for word in all_words:
        if word == target_word.lower():
            word_count += 1

    return word_count / (len(all_words) + len(types))
