from konlpy.tag import Okt
import pandas as pd
import json
from gensim.models import Word2Vec

def tokenize(doc):
    okt = Okt()
    x = []
    for i in doc:
        tokens = okt.pos(str(i), norm=True, stem=True)
        y = []
        for j in tokens:
            y.append(j[0])
        x.append(y)
    return x

def nlp(input_file, output_file, delimiter='\t', header=0, key='document', label='label', vec=False):
    data = pd.read_csv(input_file, delimiter=delimiter, header=header)
    doc = data[key]
    result = tokenize(doc)
    if vec==True:
        model = Word2Vec(result, sg=1)
        model.save("word2vec.model")
    lab = tuple(data[label])
    f = open(output_file, 'w')
    for i in range(len(lab)):
        dic = {'document':result[i], 'label':lab[i]}
        j = json.dumps(dic)
        f.write(j)
        f.write('\n')
    f.close()

nlp('ratings_train.txt', 'ratings_train.json', vec=True)
nlp('ratings_test.txt', 'ratings_test.json')
