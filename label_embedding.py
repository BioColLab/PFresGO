# This file is to generate go term embedding using anc2vec
import anc2vec.train as builder
import numpy as np

es = builder.fit('./Datasets/go-basic.obo', embedding_sz=128, batch_sz=64, num_epochs=100)
np.save('./Datasets/label-embedding-128.npy', es)
print("finishing the label embedding")


