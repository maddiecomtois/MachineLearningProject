"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 22/11/2020
Text Treatment for Data
"""

import os
import nltk
import numpy as np

directory = './Datasets'
familiar_tags = ['du', 'Du', 'dich', 'ihr', 'dir']
formal_tags = ['Sie', 'Ihnen', 'Herr', 'Frau']
convo_type_tags = ["customer service", "interview", "meeting", "with friends", "hotel reservation", "hotel check in",
                   "conversation with officer", "ordering at restaurant", "with colleagues", "convo with spouse",
                   "travel", "introductions", "with stranger", "with family", "with students", "with elderly"]


def get_feature_matrix():
    convo_dataset = []
    total_familiar = 0
    total_formal = 0
    print("Collecting Data from Dataset...")
    # loop through all the convo files in the Datasets directory
    for filename in sorted(os.listdir(directory)):
        familiar_count = 0.0
        formal_count = 0.0
        dataset_entry = []

        with open(directory + '/' + filename, 'r') as f:
            # segment the contents into individual words
            contents = nltk.word_tokenize(f.read())

            # count the number of familiar/formal words
            for word in contents:
                if word in familiar_tags:
                    familiar_count += 1
                if word in formal_tags:
                    formal_count += 1

            # get frequency of formal/familiar out of total number of total counts
            total_count = formal_count + familiar_count
            print(filename)
            print('\tNumber of familiar: ', familiar_count)
            print('\tNumber of formal: ', formal_count)
            if total_count > 0:
                print("\tFamiliar frequency: ", familiar_count, " / ", total_count, " = ", familiar_count / total_count)
                print("\tFormal frequency: ", formal_count, " / ", total_count, " = ", formal_count / total_count)

            # initialise label of the entry & append to feature list/vector
            if familiar_count > formal_count:
                total_familiar += 1
                entry_label = -1
            else:
                total_formal += 1
                entry_label = 1

            dataset_entry.append(formal_count)
            dataset_entry.append(familiar_count)

            # Get last line which contains the conversation type of this entry
            f.seek(0)
            type_tag = f.read().splitlines()[-1]
            if type_tag not in convo_type_tags:
                type_tag = "dummy_cat"
            else:
                print("\tType: ", type_tag)

            # use bag of words to create vectors for the conversation type feature
            for word in convo_type_tags:
                if type_tag == word:
                    dataset_entry.append(1)
                else:
                    dataset_entry.append(0)

            dataset_entry.append(entry_label)
            convo_dataset.append(dataset_entry)

    convo_dataset = np.array(convo_dataset)
    print("Number of data entries: ", len(convo_dataset))
    print("Total number of formal: ", total_formal)
    print("Total number of familiar: ", total_familiar)
    print("\nFeature Matrix\n", convo_dataset)
    return convo_dataset


get_feature_matrix()
