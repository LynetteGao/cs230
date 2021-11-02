import pandas as pd

import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths
TEST_CSV = "./tsv/test.tsv"
M = './SiameseLSTM.h5'

model = tf.keras.models.load_model(M, custom_objects={'ManDist': ManDist})
model.summary()

print(model.metrics_names)

plt.subplot(211)
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

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

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --



prediction = model.predict([X_test['left'], X_test['right']])
submission = pd.DataFrame()
submission['test_id'] = test_df['id']
submission['is_duplicate'] = prediction
submission.to_csv('submission_siamese1.csv',index=False)

print(prediction)