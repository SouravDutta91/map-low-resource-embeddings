import re
import json
import pickle as pkl
import multiprocessing
import requests


# Preprocessing the data

def clean_token(token):
    token = re.sub(r'[)(}{)\[\]:;,`.!?*+=#\-\'\"0-9/\\&$]', '', token)
    return token.strip()


# Load the data from corpus file,
# Remove empty words after preprocessing

def load_data(path):
    data = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence = []
            for word in line.lower().split():
                word = clean_token(word)
                if word != '':
                    sentence.append(clean_token(word))
            if len(sentence) != 0:
                data.append(sentence)

    print('Data loaded.')

    return data


# Extract the words from a given sentence

def get_words(sentences):
    words = set()
    for sentence in sentences:
        for word in sentence:
            words.add(word.lower())
    return list(words)


# Create the Word2Vec model using the parameters
# and Gensim library

def create_w2v(data,
               emb_dim=300,
               window=5,
               min_count=5,
               negative=5,
               iterations=10):

    from gensim.models import Word2Vec as w2v
    workers = multiprocessing.cpu_count()

    w2v = w2v(data,
              size=emb_dim,
              window=window,
              min_count=min_count,
              negative=negative,
              iter=iterations,
              workers=workers)

    print('Word2Vec model created.')
    return w2v


# Save the embedding file to the local disk

def prepare_embeddings(emb_file_path, w2v_file, words):
    with open(emb_file_path, 'w') as file:
        count = 0
        total_string = ''
        for word_en in words:
            if word_en in w2v_file:
                count += 1
                string = word_en + ' '
                emb = ' '.join(str(x) for x in w2v_file.wv[word_en])
                total_string += string + emb + '\n'
        total_string = str(count) + ' 300\n' + total_string
        file.write(total_string)
    print('Embeddings file saved to disk.')


# Save data to local disk using pickle

def save_pickle_file(data, file_path):
    with open(file_path, 'wb') as file:
        pkl.dump(data, file)
    print('Saved to local disk.')


# Load data from local disk using pickle

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pkl.load(file)


# Translate text from source language to target language
# using the Yandex Translation API

def translate(text, source, target):
    api_url = "https://translate.yandex.net/api/v1.5/tr.json/translate"

    # Please use your own Yandex translation API key
    api_key = "trnsl.1.1.20190814T074325Z.71d0ce74b886d4c0.9f7b4bfa310d588e981d269a184b98cc3ab9db6b"

    data = {'key': api_key,
            'text': text,
            'lang': source + '-' + target}
    translated = json.loads(requests.post(url=api_url, data=data).text)
    return translated
