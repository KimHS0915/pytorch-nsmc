from konlpy.tag import Okt
import pandas as pd
import json

def tokenize(doc):
    okt = Okt()
    x = []
    for i in doc:
        tokens = okt.pos(i, norm=True, stem=True)
        y = []
        for j in tokens:
            y.append(j[0])
        x.append(y)
    return x

def nlp(input_file, output_file, delimiter='\t', header=0, key='document', label='label'):
    data = pd.read_csv(input_file, delimiter=delimiter, header=header)
    doc = data[key][:10]
    print(len(doc))
    result = tokenize(doc)
    print(len(result))
    lab = tuple(data[label][:10])
    print(len(lab))
    f = open(output_file, 'w')
    for i in range(len(lab)):
        dic = {'document':result[i], 'label':lab[i]}
        j = json.dumps(dic)
        f.write(j)
        f.write('\n')
    f.close()

nlp('ratings_train.txt', 'ratings_train.json')
nlp('ratings_test.txt', 'ratings_test.json')
