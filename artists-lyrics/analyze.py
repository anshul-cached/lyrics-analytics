import pandas as pd
import numpy as np
import gensim.models.word2vec as w2v
import multiprocessing
import os
import re
import pprint
import sklearn.manifold
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from os import path
from PIL import Image


def createDataFrame(fileNames):
	songs = pd.read_json(fileNames, encoding='utf-8')	
	filter = songs["lyrics"] != ""
	songs=songs[filter]
	return songs


def createCorpus(songs):
	corpus = []
	for song in songs['lyrics']:
		words = song.lower().split()
		corpus.append(words)
	return corpus


def wordCloud(corpus):
	flatten=sum(corpus, [])
	wordsString=" ".join(flatten)
	wordcloud = WordCloud(max_words=2000).generate(wordsString)
	# g=wordcloud.to_image()
	# g.show()
	return wordcloud

# Ed-sheeran wordcloud
STOPWORDS.update(('oh','na na','la la','la','oh oh','well'))

df1=createDataFrame("ed-sheeran.json")
corpus1=createCorpus(df1)
wc1=wordCloud(corpus1)



# Taylor Swift wordcloud
df2=createDataFrame("taylor-swift.json")
corpus2=createCorpus(df2)
wc2=wordCloud(corpus2)


df3=df1.append(df2)
text_corpus=createCorpus(df3)



num_features = 150

min_word_count = 1

num_workers = multiprocessing.cpu_count()

context_size = 7

downsampling = 1e-1

seed = 2

songs2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

songs2vec.build_vocab(text_corpus)

songs2vec.train(text_corpus,songs2vec.corpus_count,epochs=songs2vec.iter)
if not os.path.exists("trained"):
    os.makedirs("trained")



songs2vec.save(os.path.join("trained", "songs2vec.w2v"))




import sklearn
def songVector(row):
    vector_sum = 0
    words = row.lower().split()
    for word in words:
        vector_sum = vector_sum + songs2vec[word]
    vector_sum = vector_sum.reshape(1, -1)
    normalised_vector_sum = sklearn.preprocessing.normalize(vector_sum)
    return normalised_vector_sum

df3['lyrics_vector'] = df3['lyrics'].apply(songVector)



songs_vector = []


for song_vector in df3['lyrics_vector']:
    songs_vector.append(song_vector)




X = np.array(songs_vector).reshape((144, 150))

tsne = sklearn.manifold.TSNE(n_components=2, n_iter=500, random_state=0, verbose=2)

all_word_vectors_matrix_2d = tsne.fit_transform(X)

df=pd.DataFrame(all_word_vectors_matrix_2d,columns=['X','Y'])

df.head(10)

df.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)

two_dimensional_songs = pd.concat([df3, df], axis=1)

two_dimensional_songs.head()




def name(row):
    return (row.split("/")[-1].split(".")[0])

two_dimensional_songs['song'] = df3['url'].apply(name)




from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

A = two_dimensional_songs['X']
B = two_dimensional_songs['Y']
C=two_dimensional_songs['artist']

plt.scatter(A,B)
for X,Y,Z in zip(A,B,two_dimensional_songs["song"]):                                       # <--
	ax.text(X,Y,str(Z),size=6)

plt.grid()
plt.show()


