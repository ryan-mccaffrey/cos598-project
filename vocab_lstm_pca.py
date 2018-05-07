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

# Model Params
T = 1
N = 1

################################################################################
# Initialize GloVe
################################################################################

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

# load glove
vocab, embd = loadGloVe(glove)
vocab.append("<pad>")

# preprocess embedding matrix
embedding_dim = len(embd[0])
embedding = np.vstack((np.asarray(embd, dtype=np.float32), np.zeros(embedding_dim)))

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

def embedding_layer(text_seq_batch, num_vocab, embed_dim):
    with tf.variable_scope('word_embedding'), tf.device("/cpu:0"):
        embedding_mat = tf.get_variable("embedding", [num_vocab, embed_dim], trainable=False)
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, text_seq_batch)
    return embedded_seq

# generate vectors and words from learnt embedding
def vectorizeLearntEmbd(args):
    if args.checkpoint == '':
        # Network
        if args.savefile == "det":
            vocab_size = 8803
            embedding_dim = 1000
            vocab_file = './exp-referit/data/vocabulary_referit.txt'
            vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
            pretrained_model = './exp-referit/tfmodel/referit_fc8_det_iter_25000.tfmodel'
        else:
            vocab_size = len(vocab)
            embedding_dim = len(embd[0])
            vocab_dict = dict()
            for i in range(len(vocab)): vocab_dict[vocab[i]] = i
            pretrained_model = './coco/tfmodel/cls_coco_glove_20000.tfmodel'

        # Inputs
        text_seq_batch = tf.placeholder(tf.int32, [T, N])
        embedem = embedding_layer(text_seq_batch, vocab_size, embedding_dim)  

        # Load pretrained model
        snapshot_restorer = tf.train.Saver(None)
        sess = tf.Session()
        snapshot_restorer.restore(sess, pretrained_model)     

        # Initialize arrays
        vectors = list()
        text_seq_val = np.zeros((T, N), dtype=np.int32)

        # Generate vector embeddings
        count = 0
        for word in words: 
            count += 1
            if count % 100 == 0: print("%d out of %d words processed" % (count, len(words)))

            # Preprocess word
            text_seq = text_processing.preprocess_sentence(word, vocab_dict, T)
            text_seq_val[:, 0] = text_seq

            # Extract LSTM language feature
            embedded_seq = sess.run(embedem, feed_dict={text_seq_batch:text_seq_val})
            temp = np.squeeze(np.transpose(embedded_seq))
            vectors.append(temp)

            if count == vector_count: break

        # Save vectors for easy recovery
        backup = args.savefile + "_TSNE_backup.npz"
        np.savez(os.path.join(plot_dir, backup), words=words, vectors=vectors)

    else:
        # Load saved vectors
        npzfile = np.load(os.path.join(plot_dir, args.checkpoint))
        vectors = npzfile['vectors']

    return vectors

################################################################################
# Main
################################################################################

def main(args):
    if args.glove:
        print("Embedding dimensions:", np.shape(embedding))
        vectors = list()
        adj_words = list()

        # collect word embedding for every word
        count = 0
        for word in words: 
            count += 1
            if count % 100 == 0: print("%d out of %d words processed" % (count, len(words)))

            if word in vocab:
                embd_val = embedding[vocab.index(word)]
                vectors.append(embd_val)
                adj_words.append(word)
            else: 
                print(word)

            if count == vector_count: break
    else:
        vectors = vectorizeLearntEmbd(args)    
    
    if args.k_nearest <= 0:
        if not args.glove:
            # Perform tSNE on LSTM vectors
            X_embedded = TSNE(n_components=2).fit_transform(vectors)
            plot_with_labels(X_embedded, words[:len(vectors)], args.savefile+"_tSNE.png")

        else: 
            # Perform tSNE on word embeddings
            X_embedded = TSNE(n_components=2).fit_transform(vectors)
            plot_with_labels(X_embedded, adj_words[:len(vectors)], args.savefile+"_tSNE_glove_orig.png")

    else:
        # Setup KDTree algorithm for nearest neighbors
        tree = spatial.KDTree(vectors)
        with open("testfile.txt","w") as file:
            file.write("")

        # Find k nearest neighbors for every word.
        for vec_index in range(len(vectors)):
            dist, indxs = tree.query(vectors[vec_index], k=args.k_nearest)
            results_str = words[vec_index] + ":"
            for i in indxs:
                results_str = results_str + " " + words[i]  
            with open("testfile.txt","a") as file:
                file.write(results_str + "\n")
                file.write(str(sum(vectors[vec_index]-vectors[0])) + "\n")

'''
Sample tSNE execution: 
python vocab_lstm_pca.py cls --custom

Sample k-nearest execution:
python vocab_lstm_pca.py cls_crop 5 --custom -o cls_crop_tSNE_backup.npz
'''
DESCRIPTION = """Language vector analysis suite: tSNE and k-nearest."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('savefile', help='Prefix of file to save the final plot in.') 
    parser.add_argument('k_nearest', type=int, nargs='?', default=-1)
    parser.add_argument('--glove', dest='glove', action='store_true')
    parser.add_argument('--custom', dest='glove', action='store_false')
    parser.add_argument('-o', '--checkpoint', nargs='?', default='', help='Checkpointed vectors npz file.')  
    args = parser.parse_args()
    main(args)
