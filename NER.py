#!/usr/bin/env python
# coding: utf-8

# ### Import Modules

# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")

import tensorflow as tf
print('Tensorflow version:', tf.__version__)


# ### Load and Explore the NER Dataset

# *Essential info about tagged entities*:
# - geo = Geographical Entity
# - org = Organization
# - per = Person
# - gpe = Geopolitical Entity
# - tim = Time indicator
# - art = Artifact
# - eve = Event
# - nat = Natural Phenomenon

# In[32]:


data = pd.read_csv("D://Named_Entity_Recognition//ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
data.head(20)


# In[33]:


print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())


# In[34]:


words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)


# In[35]:


tags = list(set(data["Tag"].values))
num_tags = len(tags)


# ### Retrieve Sentences and Corresponsing Tags

# In[36]:


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[37]:


getter = SentenceGetter(data)
sentences = getter.sentences


# In[38]:


sentences[0]


# ### Define Mappings between Sentences and Tags

# In[39]:


word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[40]:


word2idx, tag2idx


# ### Padding Input Sentences and Creating Train/Test Splits

# In[41]:


plt.hist([len(s) for s in sentences], bins=50)
plt.show()


# In[46]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=num_tags) for i in y]


# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ### Build and Compile a Bidirectional LSTM Model

# In[48]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional


# In[49]:


input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=50, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)
model.summary()


# In[50]:


model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# ###  Train the Model

# In[51]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback


# In[52]:


y = np.array(y_train)
y.shape


# In[53]:


get_ipython().run_cell_magic('time', '', '\nchkpt = ModelCheckpoint("model_weights.h5", monitor=\'val_loss\',verbose=1, save_best_only=True, save_weights_only=True, mode=\'min\')\n\nearly_stopping = EarlyStopping(monitor=\'val_accuracy\', min_delta=0, patience=1, verbose=0, mode=\'max\', baseline=None, restore_best_weights=False)\n\ncallbacks = [PlotLossesCallback(), chkpt, early_stopping]\n\nhistory = model.fit(\n    x=x_train,\n    y=np.array(y_train),\n    validation_split=0.2,\n    batch_size=32, \n    epochs=10,\n    callbacks=callbacks,\n    verbose=1\n)')


# ### Evaluate Named Entity Recognition Model

# In[55]:


model.evaluate(x_test, np.array(y_test))


# In[61]:


i = np.random.randint(0, x_test.shape[0]) 
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = np.argmax(np.array(y_test), axis=-1)[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))


# In[ ]:




