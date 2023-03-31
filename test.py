import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import spacy
from collections import Counter
import os
import random

nlp = spacy.load("en_core_web_sm")

base_url = "https://openlibrary.org/works/"
book_urls = [
    ("OL45883W", "Forrest_Gump.txt"),
    ("OL574455W", "The_Fellowship_of_the_Ring.txt"),
    ("OL18470W", "To_Kill_a_Mockingbird.txt"),
    ("OL28883W", "The_Hobbit.txt"),
    ("OL18661W", "The_Catcher_in_the_Rye.txt"),
    ("OL23222180M", "Harry_Potter_and_the_Philosopher_s_Stone.txt"),
    ("OL32115767M", "The_Hunger_Games.txt"),
    ("OL23959622M", "The_Da_Vinci_Code.txt"),
    ("OL23222224M", "The_Lord_of_the_Rings.txt"),
    ("OL23222240M", "Harry_Potter_and_the_Deathly_Hallows.txt"),
    ("OL23222218M", "Harry_Potter_and_the_Order_of_the_Phoenix.txt"),
    ("OL23222209M", "Harry_Potter_and_the_Half_Blood_Prince.txt"),
    ("OL32115473M", "The_Maze_Runner.txt"),
    ("OL23222205M", "Harry_Potter_and_the_Goblet_of_Fire.txt"),
    ("OL23222211M", "Harry_Potter_and_the_Prisoner_of_Azkaban.txt")
]

# Combine all sentences from all books into one list
sentences = []
for book_url in book_urls:
    book_id, book_file = book_url
    url = base_url + book_id + "/" + book_file
    response = requests.get(url, stream=True) # set stream=True to enable streaming
    for line in response.iter_lines():
        if line: # filter out keep-alive new lines
            sentences.append((book_file, line.decode('utf-8').strip()))

# Shuffle the sentences randomly
random.shuffle(sentences)

# Split the data into training and testing sets with an 80-20 split
train_size = int(0.8 * len(sentences))
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([sentence[1] for sentence in train_sentences])

train_sequences = tokenizer.texts_to_sequences([sentence[1] for sentence in train_sentences])
test_sequences = tokenizer.texts_to_sequences([sentence[1] for sentence in test_sentences])

max_length = max([len(seq) for seq in train_sequences + test_sequences])
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding="post")
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding="post")

"""model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
    tf.keras.layers.LSTM(units=64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=len(tokenizer.word_index)+1, activation="sigmoid"))
])"""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=len(tokenizer.word_index)+1, activation="sigmoid"))
])

######
learning_rate = 0.001
batch_size = 32

model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])

train_target = [to_categorical(seq, num_classes=len(tokenizer.word_index)+1) for seq in train_sequences]
train_target = pad_sequences(train_target, maxlen=max_length, padding="post")
train_target = train_target.reshape(train_target.shape[0], train_target.shape[1], len(tokenizer.word_index)+1)

test_target = [to_categorical(seq, num_classes=len(tokenizer.word_index)+1) for seq in test_sequences]
test_target = pad_sequences(test_target, maxlen=max_length, padding="post")
test_target = test_target.reshape(test_target.shape[0], test_target.shape[1], len(tokenizer.word_index)+1)

model.fit(padded_train_sequences, train_target, batch_size=batch_size, epochs=3)
model.save("book_rnn_model.h5")
########




entity_counter = Counter()

for sentence in sentences:
    # Parse the sentence with spaCy
    doc = nlp(sentence[1])
    # Loop through each named entity in the sentence
    for ent in doc.ents:
        # If the named entity is a person, increment its count in the counter
        if ent.label_ == "PERSON":
            entity_counter[ent.text] += 1


main_character = entity_counter.most_common(1)[0][0]

print("The main character is:", main_character)
