"""
@authors Madeleine Comtois and Ciara Gilsenan
@version 22/11/2020
Text Treatment for Data
"""

from nltk.tokenize import sent_tokenize, word_tokenize
example_text = "Hallo Guten Tag Grüß dich Wie geht’s? Guten Morgen. Guten Tag. Guten Abend. Willkommen in Deutschland. Hallo, wie geht’s dir? Hallo, Caroline. Hallo, Christopher, wie geht’s dir? Danke gut, und dir? Danke, auch gut. (Setz’ dich doch.) Hallo, wie geht’s dir? Danke gut, und dir? Danke, mir auch. Hallo, Ina. Hallo, Angela, wie geht’s dir? Danke gut, und dir? Danke, mir auch. (Sollen wir einen Kaffee trinken gehen?) Tschüs!"
#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))

from nltk.corpus import stopwords #note: du/Sie/ihnen are stop words!
stop_words = set(stopwords.words("german"))
for i in stop_words:
    print(i)