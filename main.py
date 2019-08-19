import sys

from utilities import load_data,\
    create_w2v,\
    save_pickle_file,\
    load_pickle_file,\
    prepare_embeddings,\
    get_words,\
    translate

# Replace this path with the path of your local project directory
PATH = '/'

#########
# NOTE: #
#########
# Please run each of the steps below separately.
# Uncomment the code for the language you want to process.
# If you run the entire script together, comment
# the lines of code marked as 'Optional'.


# STEP 1:
# Load and preprocess data, save to local disk
# --------------------------------------------


en = PATH + 'et-en/europarl-v7.et-en.en'
et = PATH + 'et-en/europarl-v7.et-en.et'

# en = PATH + 'sl-en/europarl-v7.sl-en.en'
# sl = PATH + 'sl-en/europarl-v7.sl-en.sl'

# en = PATH + 'sk-en/europarl-v7.sk-en.en'
# sk = PATH + 'sk-en/europarl-v7.sk-en.sk'

# en = PATH + 'hu-en/europarl-v7.hu-en.en'
# hu = PATH + 'hu-en/europarl-v7.hu-en.hu'

sentences_en = load_data(en)
save_pickle_file(sentences_en, PATH + 'en_data.obj')

sentences_et = load_data(et)
save_pickle_file(sentences_et, PATH + 'et_data.obj')

# sentences_sl = load_data(sl)
# save_pickle_file(sentences_sl, PATH + 'sl_data.obj')

# sentences_sk = load_data(sk)
# save_pickle_file(sentences_sk, PATH + 'sk_data.obj')

# sentences_hu = load_data(hu)
# save_pickle_file(sentences_hu, PATH + 'hu_data.obj')


# STEP 2:
# Create and save the Word2Vec models to local disk
# -------------------------------------------------


# sentences_en = load_pickle_file(PATH + 'en_data.obj')     # optional
# sentences_et = load_pickle_file(PATH + 'et_data.obj')     # optional
# sentences_sl = load_pickle_file(PATH + 'sl_data.obj')     # optional
# sentences_sk = load_pickle_file(PATH + 'sk_data.obj')     # optional
# sentences_hu = load_pickle_file(PATH + 'hu_data.obj')     # optional

w2v_en = create_w2v(sentences_en)
save_pickle_file(w2v_en, PATH + 'w2v_en.obj')

w2v_et = create_w2v(sentences_et)
save_pickle_file(w2v_et, PATH + 'w2v_et.obj')

# w2v_sl = create_w2v(sentences_sl)
# save_pickle_file(w2v_sl, PATH + 'w2v_sl.obj')

# w2v_sk = create_w2v(sentences_sk)
# save_pickle_file(w2v_sk, PATH + 'w2v_sk.obj')

# w2v_hu = create_w2v(sentences_hu)
# save_pickle_file(w2v_hu, PATH + 'w2v_hu.obj')


# STEP 3:
# Extract words from sentences
# ----------------------------


words_en = get_words(sentences_en)
words_et = get_words(sentences_et)
# words_sl = get_words(sentences_sl)
# words_sk = get_words(sentences_sk)
# words_hu = get_words(sentences_hu)

print('\nNumber of EN corpus words:', len(words_en))
print('\nNumber of ET corpus words:', len(words_et))
# print('\nNumber of SL corpus words:', len(words_sl))
# print('\nNumber of SK corpus words:', len(words_sk))
# print('\nNumber of HU corpus words:', len(words_hu))


# STEP 4:
# Prepare embedding files
# -----------------------


# w2v_en = load_pickle_file(PATH + 'w2v_en.obj')            # optional
# w2v_et = load_pickle_file(PATH + 'w2v_et.obj')            # optional
# w2v_sl = load_pickle_file(PATH + 'w2v_sl.obj')            # optional
# w2v_sk = load_pickle_file(PATH + 'w2v_sk.obj')            # optional
# w2v_hu = load_pickle_file(PATH + 'w2v_hu.obj')            # optional

emb_en = PATH + 'emb-en.txt'
prepare_embeddings(emb_en, w2v_en, words_en)

emb_et = PATH + 'emb-et.txt'
prepare_embeddings(emb_et, w2v_et, words_et)

# emb_sl = PATH + 'emb-sl.txt'
# prepare_embeddings(emb_sl, w2v_sl, words_sl)

# emb_sk = PATH + 'emb-sk.txt'
# prepare_embeddings(emb_sk, w2v_sk, words_sk)

# emb_hu = PATH + 'emb-hu.txt'
# prepare_embeddings(emb_hu, w2v_hu, words_hu)


# STEP 5:
# Prepare the dictionary
# Please note that there is a daily limit for using the translation service API
# -----------------------------------------------------------------------------


SOURCE = 'en'
TARGET = 'et'
count = 0
for word_en in words_en:
    translated = translate(word_en, SOURCE, TARGET)
    if 'text' in translated.keys():
        count += 1
        for word_trans in translated['text']:
            with open(PATH + 'en-et-dict.txt', 'a') as file:
                file.write(word_en.lower() + ' ' + word_trans.lower() + '\n')
        print(count)


# STEP 6:
# Prepare train and test data (dictionaries)
# We here consider 70% train data and 30% test data split
# -------------------------------------------------------


TRAIN_TEST_SPLIT = 0.3                                      # this can be changed
en_words_list = []
trans_words_list = []

with open(PATH + 'en-et-dict.txt', 'r') as file:
    for line in file:
        words = line.split()
        if len(words) == 2:
            en_words_list.append(words[0])
            trans_words_list.append(words[1])

total_words_freq = len(en_words_list)
split = total_words_freq - round(TRAIN_TEST_SPLIT * total_words_freq)

print('\nTotal words in dictionary: %d' % total_words_freq)
print('\nTotal words in train dictionary: %d' % split)
print('\nTotal words in test dictionary: %d' % (total_words_freq - split))

index = 0
with open(PATH + 'en-et-dict-train.txt', 'a') as train_file:
    while index < split:
        train_file.write(en_words_list[index] + ' ' + trans_words_list[index] + '\n')
        index += 1

    print('\nTrain dictionary saved.')

with open(PATH + 'en-et-dict-test.txt', 'a') as test_file:
    while index < total_words_freq:
        test_file.write(en_words_list[index] + ' ' + trans_words_list[index] + '\n')
        index += 1

    print('\nTest dictionary saved.')


sys.exit()
