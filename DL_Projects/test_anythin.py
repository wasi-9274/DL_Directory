from keras.preprocessing.text import Tokenizer

t = Tokenizer(filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n') # all without .
text = "Tomorrow will be cold."
text = text.replace(".", " .")
t.fit_on_texts([text])
print(t.word_index)
