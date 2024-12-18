# -*- coding: utf-8 -*-
"""DataPreprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11Vxs3Uvkhp6aT6i3yHIbcXAHZPOATTiy

# Data Preprocessing
"""

CROHME_PATH = './data/crohme/'
EMNIST_PATH = './data/emnist/'

test_processed = EMNIST_PATH + 'processed_balanced_test.csv'
test_raw = EMNIST_PATH + 'emnist-balanced-test.csv'

train_processed = EMNIST_PATH + 'processed_balanced_train.csv'
train_raw = EMNIST_PATH + 'emnist-balanced-train.csv'

mapping_processed = EMNIST_PATH + 'processed-mapping.csv'
mapping_raw = EMNIST_PATH + 'emnist-balanced-mapping.txt'

"""## Remove duplicated images from CROHME dataset"""

import imageio
import numpy as np
import os

images_path = CROHME_PATH

# source - https://www.kaggle.com/xainano/handwrittenmathsymbols/discussion/85277
def deletefromfolder(path):
    datadir = path
    print('Directory:', datadir)

    rmmap = dict()
    total = 0
    repeatcnt = 0
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            total += 1
            if filename.endswith('.jpg') and not filename.startswith('._'):
                filei = os.path.join(root, filename)
                imi = imageio.imread(filei)
                npi = np.asarray(imi).reshape(1, -1).reshape((2025, ))
                idf = npi.tolist()
                for i in range(len(idf)):
                    idf[i] = str(idf[i])
                strlist = ''.join(idf)

                if strlist in rmmap.keys():
                    repeatcnt += 1
                    rmmap[strlist].append(filename)
                else:
                    rmmap[strlist] = list()

    #for key in rmmap:
    #    print(rmmap[key])
    print('Repeat/Total: {}/{}'.format(repeatcnt, total))
    for key in rmmap:
        for item in rmmap[key]:
            os.remove(os.path.join(datadir, item))

dirlist = os.listdir(CROHME_PATH)
for item in dirlist:
    deletefromfolder(os.path.join(images_path,item))

"""## Mapping"""

import pandas as pd

extra_chars = ['(', ')', '+', '-', '=']
df = pd.read_csv(mapping_raw, sep = ' ', header=None, names=["id", "code"])
chars = []
for x in df.code:
    chars.append(chr(x))

df["char"] = chars
nextId = df.shape[0]
for i in range(nextId, nextId + len(extra_chars)):
    c = extra_chars[i - nextId]
    df.loc[i] = [i, ord(c), c]

df.to_csv(mapping_processed, index=False)

"""## EMNIST data preprocessing"""

import numpy as np
import pandas as pd
import os

def mirror(X):
    res = np.zeros(X.shape)
    n = 28
    for r in range(n, n**2 + 1, n):
        l = r - n
        for k in range(l, r):
            index = l + (r - k - 1)
            res[:,k] = X[:,index]

    return res

def rotate_clockwise(X):
    res = np.zeros(X.shape)
    size = 28
    k = 0
    for i in reversed(range(size)):
        j = i
        while j < size**2:
            res[:,k] = X[:,j]
            k += 1
            j += size

    return res

def rotate(X, times):
    for i in range(times):
        X = rotate_clockwise(X)

    return X

# TODO: remove one confusing character from pairs such as "0" and "O", "I", "l" and "1"
def process_data(file_from, file_to):
    label = 'label'
    names = [label] + ["px" + str(i) for i in range(784)]
    data = pd.read_csv(file_from, header=None, names=names)

    Y_data = data[label]
    X_data = data.drop(labels = [label], axis = 1)

    X_data = X_data / 255
    X_data = np.where(X_data > 0.5, 1, 0)
    X_data = rotate(X_data, times=3)
    X_data = mirror(X_data)

    data = pd.DataFrame(X_data, columns=names[1:], dtype='int')
    data.insert(0, label, Y_data)
    data.to_csv(file_to, index=False)

process_data(train_raw, train_processed)
process_data(test_raw, test_processed)

"""## Image transformation to EMNIST format"""

import os
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from random import sample

df = pd.read_csv(mapping_processed)
char2code = {}
for index, row in df.iterrows():
    char2code[row['char']] = row['id']

import cv2

def img2emnist(filepath, char_code):
    img = cv2.imread(filepath, 0)
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.erode(img, kernel, iterations = 1)

    img = Image.fromarray(dilation).resize((28, 28))
    inv_img = ImageOps.invert(img)

    flatten = np.array(inv_img).flatten()
    flatten = flatten / 255
    flatten = np.where(flatten > 0.5, 1, 0)

    csv_img = ','.join([str(num) for num in flatten])
    csv_str = '{},{}'.format(char_code, csv_img)
    return csv_str

train_size = 2400
test_size = 400

f_test = open(test_processed, 'a')
f_train = open(train_processed, 'a')

for c in extra_chars:
    print('Processing "{}" character...'.format(c))
    current_dir = CROHME_PATH + c + '/'
    files = [f for r, d, f in os.walk(current_dir)]
    subset = sample(files[0], train_size + test_size)
    train_subset = subset[0:train_size]
    test_subset = subset[train_size:train_size + test_size]

    for filename in train_subset:
        csv_str = img2emnist(current_dir + filename, char2code[c])
        print(csv_str, file=f_train)

    for filename in test_subset:
        csv_str = img2emnist(current_dir + filename, char2code[c])
        print(csv_str, file=f_test)

f_test.close()
f_train.close()