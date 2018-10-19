from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from pickle import dump


#Loading document into the memory
def load_text_doc(filename):
	file = open(filename,'r')
	text = file.read()
	file.close()

	return text

def clean_text_doc(doc):
	#Splitting documents by white spaces
	tokens = doc.split()
	#Removing punctuation from each token
	table = str.maketrans('','',punctuation)
	tokens = [w.translate(table) for w in tokens]
	#removing non alphabetic tokens
	tokens = [word for word in tokens if word.isalpha()]
	stop_words = set(stopwords.words('english'))
	#Removing stop words
	tokens = [word for word in tokens if not word in stop_words]
	#removing short tokens
	tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)

	return tokens

def process_docs(directory,is_train):
	documents = list()

	for filename in listdir(directory):
		#skipping all the reviews in test set
		if(is_train and filename.startswith('cv9')):
			continue

		if(not is_train and not filename.startswith('cv9')):
			continue

		path = directory + '/' +filename
		doc = load_text_doc(path)
		tokens = clean_text_doc(doc)
		documents.append(tokens)

	return documents

def save_dataset(dataset,filename):
	dump(dataset,open(filename,'wb'))
	print("Saved:%s" %filename)


def main():

	neg_doc = process_docs('txt_sentoken/neg',True)
	pos_doc = process_docs('txt_sentoken/pos',True)

	trainX = neg_doc + pos_doc
	#O for negative review and 1 for positive review
	trainY = [0 for _ in range(900)] + [1 for _ in range(900)]
	save_dataset([trainX,trainY],'train.pkl')

	neg_doc = process_docs('txt_sentoken/neg',False)
	pos_doc = process_docs('txt_sentoken/pos',False)

	trainX = neg_doc + pos_doc
	#O for negative review and 1 for positive review
	trainY = [0 for _ in range(100)] + [1 for _ in range(100)]
	save_dataset([trainX,trainY],'test.pkl')


main()
