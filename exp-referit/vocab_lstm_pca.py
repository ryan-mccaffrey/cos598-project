from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import tensorflow as tf
import argparse

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from models import text_objseg_model as segmodel
from models import vgg_net, lstm_net, processing_tools
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util import text_processing

################################################################################
# Parameters
################################################################################

vocab_file = './exp-referit/data/vocabulary_referit.txt'
pretrained_model = './exp-referit/tfmodel/referit_fc8_det_iter_25000.tfmodel'
plot_dir = 'tSNE'

# Model Params
T = 20
N = 1
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

D_im = 1000
D_text = lstm_dim

# np.seterr(divide='ignore', invalid='ignore')

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

def main(args):
    if args.checkpoint == '':
        ################################################################################
        # Load evaluation network
        ################################################################################

        # Inputs
        text_seq_batch = tf.placeholder(tf.int32, [T, N])
        imcrop_batch = tf.placeholder(tf.float32, [N, 224, 224, 3])
        lstm_top_batch = tf.placeholder(tf.float32, [N, D_text])
        fc8_crop_batch = tf.placeholder(tf.float32, [N, D_im])
        spatial_batch = tf.placeholder(tf.float32, [N, 8])

        # Language feature (LSTM hidden state)
        lstm_top = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)

        # Local image feature
        fc8_crop = vgg_net.vgg_fc8(imcrop_batch, 'vgg_local', apply_dropout=False)

        # L2-normalize the features (except for spatial_batch)
        # and concatenate them along axis 1 (feature dimension)
        if args.savefile == "det":
            feat_all = tf.concat(axis=1, values=[tf.nn.l2_normalize(lstm_top_batch, 1),
                                     tf.nn.l2_normalize(fc8_crop_batch, 1),
                                     spatial_batch])
        else:
            feat_all = tf.concat(axis=1, values=[tf.nn.l2_normalize(lstm_top_batch, 1),
                                 tf.nn.l2_normalize(fc8_crop_batch, 1)])

        # Outputs
        # MLP Classifier over concatenate feature
        with tf.variable_scope('classifier'):
            mlp_l1 = fc_relu('mlp_l1', feat_all, output_dim=mlp_hidden_dims)
            mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)
        scores = mlp_l2

        # Load pretrained model
        if args.savefile == "det":
            variable_name_mapping= None
            if tf.__version__.split('.')[0] == '1':
                variable_name_mapping = {
                    v.op.name.replace(
                        'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel',
                        'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix').replace(
                        'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias',
                        'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias'): v
                    for v in tf.global_variables()}
            snapshot_restorer = tf.train.Saver(variable_name_mapping)
        else:
            snapshot_restorer = tf.train.Saver(None)

        sess = tf.Session()
        snapshot_restorer.restore(sess, pretrained_model)

        ################################################################################
        # Load vocabulary
        ################################################################################

        vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

        ################################################################################
        # Testing
        ################################################################################

        # Pre-allocate arrays
        text_seq_val = np.zeros((T, N), dtype=np.int32)
        lstm_top_val = np.zeros((N, D_text))
        words = list(vocab_dict.keys())
        vectors = list()

        print("Unique words in vocabulary:", len(words))
        count = 0

        # extract LSTM feature vectors per word
        for word in words: 
            count += 1
            if count % 100 == 0: print("%d out of %d words processed" % (count, len(words)))

            # Form batch
            text_seq = text_processing.preprocess_sentence(word, vocab_dict, T)
            text_seq_val[:, 0] = text_seq

            # Extract language feature
            lstm_top_val[...] = sess.run(lstm_top, feed_dict={text_seq_batch:text_seq_val})
            temp = np.squeeze(np.transpose(lstm_top_val))
            # print(np.shape(temp)); exit()
            vectors.append(temp)

            if count == 100: break

        # Save TSNE vectors for easy recovery
        backup = args.savefile + "_TSNE_backup.npz"
        np.savez(os.path.join(plot_dir, backup), words=words, vectors=vectors)

    else:
        # Load saved vectors
        npzfile = np.load(os.path.join(plot_dir, args.checkpoint))
        words = npzfile['words']
        vectors = npzfile['vectors'] 

    # Perform PCA
    X_embedded = TSNE(n_components=2).fit_transform(vectors)
    plot_with_labels(X_embedded, words[:len(vectors)], args.savefile+"_TSNE_plot.png")
      

'''
Sample execution: 
python exp-referit/vocab_lstm_pca.py cls_crop -o cls_crop_SNE_backup.npz
'''
DESCRIPTION = """Perform PCA on LSTM feature representation on each word in the vocabulaty."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('savefile', help='Prefix of file to save the final plot in.') 
    parser.add_argument('-o', '--checkpoint', nargs='?', default='', help='Checkpointed vectors npz file.')  
    args = parser.parse_args()
    main(args)
