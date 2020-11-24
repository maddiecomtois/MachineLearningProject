"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 22/11/2020
Text Treatment for Data
"""

import os
import nltk

directory = '/Users/maddie/Documents/TCDModules/Machine_Learning/Project/MachineLearningProject/Datasets'
familiar_tags = ['du', 'Du', 'dich', 'ihr', 'dir']
formal_tags = ['Sie', 'Ihnen']

# loop through all the convo files in the Datasets directory
for filename in sorted(os.listdir(directory)):
    familiar_count = 0.0
    formal_count = 0.0
    file_object = open(directory + '/' + filename, "r")
    # segment the contents into individual words
    contents = nltk.word_tokenize(file_object.read())

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

