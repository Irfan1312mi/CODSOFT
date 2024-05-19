#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install keras


# In[3]:


pip install tensorflow 


# In[ ]:


conda install tensorflow


# In[ ]:


import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.utils import to_categorical

# Load the VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Load the captions and preprocess them
captions = [
    'an elephant standing in the grass',
    'a giraffe eating leaves from a tree',
    'a lion sleeping on the rock',
    'a cat sitting on the sofa',
    'a dog playing with a ball'
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
sequences = tokenizer.texts_to_sequences(captions)
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Preprocess the images and extract features
images = ['elephant.jpg', 'giraffe.jpg', 'lion.jpg', 'cat.jpg', 'dog.jpg']
features = []
for img_path in images:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    features.append(feature)
features = np.array(features)

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=256, input_length=max_length))
model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Prepare the data for training
X = features
y = to_categorical(padded_sequences)

# Train the model
model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# Save the trained model
model.save('image_captioning_model.h5')

