from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import tensorflow as tf
import argparse

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from scipy import spatial
from tensorflow.contrib import learn

from models import text_objseg_model as segmodel
from models import vgg_net, lstm_net, processing_tools
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util import text_processing

################################################################################
# Parameters
################################################################################

plot_dir = 'tSNE'
vector_count = 500

# glove file
glove = './exp-referit/data/glove.6B.50d.txt'

# load dataset vocabulary
vocab_file = './exp-referit/data/vocabulary_referit.txt'
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
words = list(vocab_dict.keys())
print("Unique words in vocabulary:", len(words))

# Model Params
T = 1
N = 1
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
D_text = lstm_dim

################################################################################
# Helper Functions
################################################################################

# plot lower dimensionality embedding
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(os.path.join(plot_dir,filename))

# load GloVe embedding matrix
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r', encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

# generate vectors and words from learnt embedding
def vectorizeLearntEmbd(args):
    if args.checkpoint == '':
        ### Load LSTM ###

        # Determine network:
        if args.savefile == "det":
            pretrained_model = './exp-referit/tfmodel/referit_fc8_det_iter_25000.tfmodel'
        else:
            pretrained_model = './exp-referit/tfmodel/model_crop.tfmodel'

        # Inputs
        text_seq_batch = tf.placeholder(tf.int32, [T, N])
        lstm_top_batch = tf.placeholder(tf.float32, [N, D_text])

        # Language feature (LSTM hidden state)
        lstm_top = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)

        # Load pretrained model
        variable_name_mapping= None
        if args.savefile == "det":
            if tf.__version__.split('.')[0] == '1':
                variable_name_mapping = {
                    v.op.name.replace(
                        'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel',
                        'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix').replace(
                        'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias',
                        'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias'): v
                    for v in tf.global_variables()}
        snapshot_restorer = tf.train.Saver(variable_name_mapping)
        sess = tf.Session()
        snapshot_restorer.restore(sess, pretrained_model)     

        ### Generate vectors ###

        # Pre-allocate arrays
        text_seq_val = np.zeros((T, N), dtype=np.int32)
        lstm_top_val = np.zeros((N, D_text))
        embd_val = np.zeros((N, embed_dim))
        lstm_vectors = list()
        embd_vectors = list()

        # extract LSTM feature lstm_vectors per word
        count = 0
        for word in words: 
            count += 1
            if count % 100 == 0: print("%d out of %d words processed" % (count, len(words)))

            # Form batch
            text_seq = text_processing.preprocess_sentence(word, vocab_dict, T)
            text_seq_val[:, 0] = text_seq

            # Extract LSTM language feature and word embedding vector
            lstm_top_val[...]  = sess.run(lstm_top, feed_dict={text_seq_batch:text_seq_val})
            temp = np.squeeze(np.transpose(lstm_top_val))
            lstm_vectors.append(temp)
            temp = np.squeeze(np.transpose(embd_val))
            embd_vectors.append(temp)

            if count == vector_count: break

        # Save TSNE lstm_vectors for easy recovery
        backup = args.savefile + "_TSNE_backup.npz"
        np.savez(os.path.join(plot_dir, backup), words=words, 
                 lstm_vectors=lstm_vectors, embd_vectors=embd_vectors)

    else:
        # Load saved lstm_vectors
        npzfile = np.load(os.path.join(plot_dir, args.checkpoint))
        lstm_vectors = npzfile['lstm_vectors']
        embd_vectors = npzfile['embd_vectors']

    return lstm_vectors, embd_vectors

################################################################################
# Main
################################################################################

def main(args):
    if args.glove:
        # load GloVe
        vocab, embd = loadGloVe(glove)
        embedding = np.asarray(embd)
        print("Embedding dimensions:", np.shape(embedding))

        embd_vectors = list()
        adj_words = list()

        # collect word embedding for every word
        count = 0
        for word in words: 
            count += 1
            if count % 100 == 0: print("%d out of %d words processed" % (count, len(words)))

            if word in vocab:
                embd_val = embd[vocab.index(word)]
                embd_vectors.append(embd_val)
                adj_words.append(word)
            else: 
                print(word)

            if count == vector_count: break
    else:
        lstm_vectors, embd_vectors = vectorizeLearntEmbd(args)    
    
    if args.k_nearest <= 0:
        if not args.glove:
            # Perform tSNE on LSTM vectors
            X_embedded = TSNE(n_components=2).fit_transform(lstm_vectors)
            plot_with_labels(X_embedded, words[:len(lstm_vectors)], args.savefile+"_TSNE_lstm.png")
        
            # Perform tSNE on word embeddings
            X_embedded = TSNE(n_components=2).fit_transform(embd_vectors)
            plot_with_labels(X_embedded, words[:len(embd_vectors)], args.savefile+"_TSNE_embd.png")

        else: 
            # Perform tSNE on word embeddings
            X_embedded = TSNE(n_components=2).fit_transform(embd_vectors)
            plot_with_labels(X_embedded, adj_words[:len(embd_vectors)], args.savefile+"_TSNE_embd_glove.png")

    else:
        # Setup KDTree algorithm for nearest neighbors
        tree = spatial.KDTree(lstm_vectors)
        with open("testfile.txt","w") as file:
            file.write("")

        # Find k nearest neighbors for every word.
        for vec_index in range(len(lstm_vectors)):
            dist, indxs = tree.query(lstm_vectors[vec_index], k=args.k_nearest)
            results_str = words[vec_index] + ":"
            for i in indxs:
                results_str = results_str + " " + words[i]  
            with open("testfile.txt","a") as file:
                file.write(results_str + "\n")
                file.write(str(sum(lstm_vectors[vec_index]-lstm_vectors[0])) + "\n")

'''
Sample tSNE execution: 
python exp-referit/vocab_lstm_pca.py det --custom -o det_tSNE_backup.npz

Sample k-nearest execution:
python exp-referit/vocab_lstm_pca.py cls_crop 5 --custom -o cls_crop_tSNE_backup.npz
'''
DESCRIPTION = """Language vector analysis suite: tSNE and k-nearest."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('savefile', help='Prefix of file to save the final plot in.') 
    parser.add_argument('k_nearest', type=int, nargs='?', default=-1)
    parser.add_argument('--glove', dest='glove', action='store_true')
    parser.add_argument('--custom', dest='glove', action='store_false')
    parser.add_argument('-o', '--checkpoint', nargs='?', default='', help='Checkpointed lstm_vectors npz file.')  
    args = parser.parse_args()
    main(args)
