import pandas as pd

import tensorflow as tf

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths
M = './SiameseLSTM2.h5'
TEST_CSV = "./tsv/test.tsv"

model = tf.keras.models.load_model(M, custom_objects={'ManDist': ManDist})
model.summary()


# Load training set
test_df = pd.read_table(TEST_CSV,header = None, names = ['id','qid1','qid2',
                                                              
                                                              'question1', 'question2','is_duplicate'])
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)
Y_test = test_df['is_duplicate'].values

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape
assert len(X_test['left']) == len(Y_test)


prediction = model.predict([X_test['left'], X_test['right']])
loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)
print(prediction)