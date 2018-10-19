from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input,Dense,Flatten,Dropout,Embedding
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers.merge import concatenate


def load_dataset(filename):
	return load(open(filename,'rb'))

def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_length(lines):
	return max([len(s.split()) for s in lines])

def encode_text(tokenizer,lines,length):
	encoded = tokenizer.texts_to_sequences(lines)
	padded = pad_sequences(encoded,maxlen=length,padding='post')
	return padded

def multichannel_model(length,vocab_size):
	#Channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size,100)(inputs1)
	conv1 = Conv1D(filters=32,kernel_size=4,activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)

	#Channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size,100)(inputs2)
	conv2 = Conv1D(filters=32,kernel_size=6,activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)

	#Channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size,100)(inputs3)
	conv3 = Conv1D(filters=32,kernel_size=8,activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)

	#merging channels
	merged = concatenate([flat1,flat2,flat3])

	#interpretation
	dense1 = Dense(10,activation='relu')(merged)
	outputs = Dense(1,activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

	#Compiling
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

	#Summarize
	print(model.summary())
	plot_model(model,show_shapes=True,to_file = 'multichannel.png')

	return model

def main():

	#Loading datasets
	trainLines,trainLabels = load_dataset('train.pkl')

	#Create Tokenizer
	tokenizer = create_tokenizer(trainLines)
	length = max_length(trainLines)

	vocab_size = len(tokenizer.word_index) + 1
	print('Max Document Length:%d' % length)
	print('Vocabulary size:%d' % vocab_size)

	#Encoding Data
	trainX = encode_text(tokenizer,trainLines,length)
	print(trainX.shape)

	model = multichannel_model(length,vocab_size)
	model.fit([trainX,trainX,trainX],array(trainLabels),epochs = 25,batch_size = 32)

	model.save('model.h5')

main()