import spacy
import requests
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

nlp = spacy.load("en_core_web_sm")

# Define the maximum length of sequences
max_length = 141
tokenizer = Tokenizer()

# Load the tokenizer from file
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# Load the trained model
model = tf.keras.models.load_model("C:/Users/vrjav/Downloads/spring 2023/DL/DL_P/book_rnn_model.h5")

# Download and preprocess the text of the new book
#new_book_url = "https://openlibrary.org/works/OL10737288M/The_Thief.txt"
new_book_url = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
new_book_text = requests.get(new_book_url).text
new_book_sentences = [(None, sent.text) for sent in nlp(new_book_text).sents]
new_book_sequences = tokenizer.texts_to_sequences([sentence[1] for sentence in new_book_sentences])
new_book_padded_sequences = pad_sequences(new_book_sequences, maxlen=max_length, padding="post")
new_book_target = model.predict(new_book_padded_sequences)

# Extract the most frequently mentioned named entity from the new book's text
new_book_entity_counter = Counter()
for sentence in new_book_sentences:
    doc = nlp(sentence[1])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            new_book_entity_counter[ent.text] += 1
new_book_main_character = new_book_entity_counter.most_common(1)[0][0]

# Find the index of the main character in the tokenizer's word index
main_char_index = tokenizer.word_index.get(new_book_main_character, 0)



# Find the index of the main character in the tokenizer's word index
main_char_index = tokenizer.word_index.get(new_book_main_character, 0)

# Iterate through the sentences and find the one containing the predicted main character
pred_main_char_sentence = ""
for sentence in new_book_sentences:
    if new_book_main_character in sentence[1]:
        pred_main_char_sentence = sentence[1]
        break

print("The predicted main character is:", new_book_main_character)
print("The sentence containing the predicted main character is:", pred_main_char_sentence)
