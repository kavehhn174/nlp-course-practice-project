import sys
import preprocess
import spellcorrection
import classification

while (True):
    print('\n')
    print("Select a feature:")
    print("1. Text Preprocessing")
    print("2. Spell Check and Dictation")
    print("3. Text Classification")
    selected_feature = input("Enter your selection (1, 2, or 3): ")

    if selected_feature == "1":
        preprocess.preprocess_text()

    elif selected_feature == "2":
        spellcorrection.spell_check()

    elif selected_feature == "3":
        classification.select_classification_text()

    else:
        print("Invalid selection")
