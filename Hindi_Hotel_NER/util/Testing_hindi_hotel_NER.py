
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import spacy
nlp = spacy.load('en_core_web_sm')

id2classes=['pad',
 'B-Amenities',
 'B-Amount',
 'B-CustomerType',
 'B-HotelName',
 'B-Location',
 'B-Nationality',
 'B-NoOfRooms',
 'B-NoOfTravellers',
 'B-RoomType',
 'B-Star_Rate',
 'B-checkin',
 'B-checkout',
 'I-Amenities',
 'I-Amount',
 'I-CustomerType',
 'I-HotelName',
 'I-Location',
 'I-Nationality',
 'I-NoOfRooms',
 'I-NoOfTravellers',
 'I-RoomType',
 'I-Star_Rate',
 'I-checkin',
 'I-checkout',
 'O']

CNN1 = tf.keras.layers.Conv1D(300, 3, activation='relu', name='CNN1', padding='same')
LSTM1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 100, 
                                                     return_sequences = True, 
                                                     recurrent_dropout = 0.5), name='Bidir1')
LSTM2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, 
                                                    return_sequences= True,
                                                    recurrent_dropout = 0.5), name='Bidir2')
DENSE = tf.keras.layers.Dense(26, activation='softmax')

emb=  tf.keras.Input((50, 300), dtype="float32")

x = CNN1(emb)
x = LSTM1(x)
intermediate = LSTM2(x)
x = DENSE(intermediate)

bar_model = tf.keras.models.Model(emb, x)

# from tensorflow.keras.models import load_weights
model1 = bar_model
model1.load_weights('/content/NER_HINDI_HOTEL.h5')

model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

import pickle
with open('/content/drive/My Drive/word_vec.pkl','rb') as f:
    word_vec = pickle.load(f)

with open('/content/drive/My Drive/tokenizer.pickle', 'rb') as handle:
    word2id = pickle.load(handle)

def prepo_string(text:str, word2id:dict=word2id, maxlen:int=50, unknown:str='unk', padding:str='pad'):
    tokens = [word2id.get(i.text, word2id.get(unknown)) for i in nlp(text)]
    tokens = tokens + ([word2id.get(padding)]*(maxlen-len(tokens)))
    return tokens[:maxlen]

def predict(text):
    list1=[]
    inp = np.asarray([prepo_string(text)])
    for i in inp:
      list1.append(word_vec[i])
    arr = np.array(list1) 
    res = model1.predict(arr)
    result = res.argmax(-1)[0]
    return [id2classes[i] for i in result]

def predict_NER(data):
  for i in data:
   r=predict(i)
   print(' ')
   for i,j in zip(i.split(), r):
        print(i.ljust(15), j)
   print('-'*50)