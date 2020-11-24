Language Register Recommendations 

Motivation:   
Many world languages incorporate a system of formal and familiar registers depending on the given situation or context that a conversation takes place.  For example, certain pronouns are used to address people of higher status than would be used with one’s peers.  However, the rules for using certain pronouns are not always clear, and for language learners it is a difficult concept to know which register is best to use in a given context.  This project therefore will analyse different German transcripts of conversations that have different contexts, grammar, and vocabulary and predict which register (Du vs Sie) is best to use given a certain input transcript.

Dataset:  
	We will be collecting our data from transcripts of conversations in German from BBC, Deutschlandfunk Kultur, and other web sources.  The data will contain several types of encounters where a percentage of registers are formal (Sie) and familiar (Du).  These will include meetings, interviews, text messaging, etc.  Each instance in the dataset will contain 4 or more columns of features and class labels.  The labels will be +1 → Formal and -1 → Familiar.  The features will include the type of encounter, i.e. meeting/interview/SMS, the frequency of familiar words, and the frequency of formal words in the instance.  These features will vary depending on the collected data.  We will pick and mark these features for all of the data we use. 

Method:    
	We will implement different feature engineering techniques to construct our data for our model.  We will use the NLTK python library and the Bag of Words technique to tokenize and evaluate our textual data.  After constructing our features we will train 80% of our data using Kernalised Support Vector Machines and Logistic Regression.  The remaining 20% will be used as testing data for the different models.


Experiments:
	For evaluation we will use cross-validation to determine the hyperparameters used for our SVM and Logistic Regression models.  We will compare the mean squared error, ROC diagrams, and confusion matrices to analyse the performance of each model and determine which is best.  We will also be comparing these performances using Naive Bayes as a baseline model.  In addition to the data used for testing the model, transcripts or articles with different content used for training, such as newspaper articles or formal reports, will be used to see how the model performs given the different input.
