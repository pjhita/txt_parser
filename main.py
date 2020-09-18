import os
import time
import logging
from bs4 import BeautifulSoup, Comment
import pandas as pd
import urllib.request
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from logdna import LogDNAHandler

############################################################################################
Key = 'insert_your_logdna_key_here'

log = logging.getLogger('logdna')
log.setLevel(logging.INFO)

options = {
    'app': 'Txt_Parser',
    'hostname': 'whitewolf-python-test',
    'ip': '86.1.53.216',
    'index_meta': True,
    'url': 'https://logs.logdna.com/logs/ingest'
}

handler = LogDNAHandler(Key, options)

log.addHandler(handler)
############################################################################################

def check_tags(var): # a function to check for tags from the text that will be pulled from html
    blacklist = ['style', 'head', 'title', 'script', '[document]'] # a blacklist of tags that I do not want in the text
    if var.parent.name in blacklist:
        return False
    if isinstance(var, Comment): # this part checks for comments on the html code as this is not wanted when pulling the text
        return False
    return True
    log.info('Removing comments from selected html...')

def get_text(page): # a function to pull the text from the url that is defined in the main funtion
    soup = BeautifulSoup(page, 'html.parser') # this puts all the html into a BeautifulSoup variable called soup
    all_text = soup.findAll(text=True) # this searches for all the raw text inside of the pulled html, thus the reason for text=True
    log.info('All raw text found.')
    clean_text = filter(check_tags, all_text) # this cleans the text through filtering all_text through the use of the blacklist defined in the function above
    log.info('Text has been cleaned via blacklist.')
    return " ".join(t.strip() for t in clean_text) # this removes the gaps inside clean_text

def tokenization(): # a function that tokenizes according to word, sentence and also removes stopwords
    global filtered_sentence
    global filtered_words
    fs = open("Tokenization-Sentence-dynamic.txt", "w", encoding="utf8")
    fw = open("Tokenization-Words-dynamic.txt", "w", encoding="utf8")
    stop_words = set(stopwords.words("english")) # these are the predefined stop words by nltk
    sentence = sent_tokenize(cleaned_text) # tokenizes cleaned_text (defined in the main function) by sentence
    words = word_tokenize(cleaned_text) # tokenizes cleaned_text (defined in the main function) by word
    filtered_sentence = [x for x in sentence if not x in stop_words and x != '\n'] # a filtered list where the text is tokenized by sentence, stop words and removal of new lines
    filtered_words = [x for x in words if not x in stop_words and x != '\n'] # a filtered list where the text is tokenized by words, stop words and removal of new lines
    with fs as output:
        output.write(str(filtered_sentence))
        output.close()
    with fw as output_2:
        output_2.write(str(filtered_words))
        output_2.close()

def speech_tagging(): # a function that speech tags each filtered word from the previous function
    global tagged_speech
    f = open("Speech_Tagged-dynamic.txt", "w", encoding="utf8")
    tagged_speech = nltk.pos_tag(filtered_words) # this takes the filtered_words and tokenizes further through default nltk speech tagging
    with f as output:
        output.write(str(tagged_speech))
        output.close()

def stemmer(): # a function that stems each word using the PorterStemmer method defined in the nltk package
    global stemmed_sentences
    f = open("Stemmed-dynamic.txt", "w", encoding="utf8")
    pstemmer = PorterStemmer()
    stem_sentence = [] # an empty list that will store the stemmed words
    for w in filtered_words: # for each word in filtered words do
        stem_sentence.append(pstemmer.stem(w)) # append the list created above with the stemmed word
        stem_sentence.append(" ") # append the list with a space
    stemmed_sentences = "".join(stem_sentence) # a variable holding the final stemmed sentences
    f.write(stemmed_sentences)
    f.close()

def tf_idf(): # a function that performs the tf.idf formula using vectors
    global df
    f = open("tf_idf-dynamic.txt", "w", encoding="utf8")
    docs = [w.lower() for w in filtered_sentence] # ensuring lowercase on each word in the filtered_sentence variable
    vectorizer = TfidfVectorizer() # using a great tool from sklearn to create a vector that stores the tf.idf values for each word
    vectors = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names() # pulls the words to be placed on the axis
    dense = vectors.todense()
    denselist = dense.tolist()
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    df = pd.DataFrame(denselist, columns=feature_names)
    f.write(df.T.to_string())
    f.close()

def tf_idf_both(): # a function that get the tf.idf calculations for both URLs
    fn1 = os.path.join(os.path.dirname(__file__), 'output_for_media_memorability/Tokenization-Words.txt')
    fn2 = os.path.join(os.path.dirname(__file__), 'output_for_siirh2020/Tokenization-Words.txt')
    f = open("tf_idf-joint.txt", "w", encoding="utf8")
    file1 = open(fn1, "r")
    file2 = open(fn2, "r")
    doc1 = file1.read().lower()
    doc2 = file2.read().lower()
    print(doc1)
    print(doc2)
    vectorizer = TfidfVectorizer() # using a great tool from sklearn to create a vector that stores the tf.idf values for each word
    vectors = vectorizer.fit_transform([doc1, doc2]) # creating a list with both docs
    feature_names = vectorizer.get_feature_names() # pulls the words to be placed on the axis
    dense = vectors.todense()
    denselist = dense.tolist()
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    df = pd.DataFrame(denselist, columns=feature_names)
    f.write(df.T.to_string())
    f.close()

def user_input(): # a function to get user input to decide which url they want to see the information retrieval take place on!
    global url
    y = None
    while y not in (1, 2):
        user_input = input("Which website would you like to perform retrieval on? 1 - Media Memorability or 2 - SIIRH2020: ")
        log.warning('User did not select correctly...')
        if user_input == "1":
            url = urllib.request.urlopen('http://www.multimediaeval.org/mediaeval2019/memorability/').read()
            log.info('User selected Media Memorability (1)')
            break
        elif user_input == "2":
            url = urllib.request.urlopen('https://sites.google.com/view/siirh2020/').read()
            log.info('User selected siirh2020 (2)')
            break

def main(): # the main function that calls all the other functions respectivily and prints the relevant outputs (outputs are provided in the zip)
    global cleaned_text
    user_input()
    cleaned_text = get_text(url)
    f1 = open('HTML_Text_Parsing-dynamic.txt', 'w', encoding="utf8")
    f1.write(cleaned_text)
    log.info('File written.')
    f1.close()
    tokenization()
    log.info('Tokenization complete.')
    speech_tagging()
    log.info('Speech tagging complete.')
    stemmer()
    log.info('Stemming complete.')
    tf_idf()
    log.info('Calculation for tf_idf successful.')
    tf_idf_both()
    log.info('Procedure completed!', options)
    print("Procedure complete. All output can be found inside the files that have been created!")

if __name__ == "__main__":
    main()
