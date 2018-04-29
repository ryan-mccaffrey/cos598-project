from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit

from random import randint, shuffle
from collections import defaultdict
from models import vgg_net, lstm_net, processing_tools
from models import text_objseg_model as segmodel
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util import im_processing, text_processing, eval_tools

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
query_file = './exp-referit/data/referit_query_test.json'
imcrop_file = './exp-referit/data/referit_imcrop.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

# Note: i believe fully trained model is iter_0.tfmodel
pretrained_model = './exp-referit/tfmodel/referit_fc8_cls_iter_%d.tfmodel'

# Model Params
T = 20
N = 1
input_H = 224
input_W = 224
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

vgg_dropout = False
mlp_dropout = False

D_im = 1000
D_text = lstm_dim

################################################################################
# Evaluation network
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N]) 
imcrop_batch = tf.placeholder(tf.float32, [N, input_H, input_W, 3])
label_batch = tf.placeholder(tf.float32, [N, 1])

# Outputs
scores = segmodel.text_objseg_cls(text_seq_batch, imcrop_batch, 
    num_vocab, embed_dim, lstm_dim, mlp_hidden_dims,
    vgg_dropout=vgg_dropout, mlp_dropout=mlp_dropout)

# Load pretrained model
variable_name_mapping= None
if tf.__version__.split('.')[0] == '1':
    variable_name_mapping = {
        v.op.name.replace(
            'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel',
            'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix').replace(
            'rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias',
            'RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias'): v
        for v in tf.global_variables()}

# snapshot_restorer = tf.train.Saver(variable_name_mapping)
snapshot_restorer = tf.train.Saver(None)
sess = tf.Session()
# restores the iter_0.tfmodel
snapshot_restorer.restore(sess, pretrained_model % 0)

################################################################################
# Load annotations
################################################################################

query_dict = json.load(open(query_file))
imcrop_dict = json.load(open(imcrop_file))
imsize_dict = json.load(open(imsize_file))
imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

################################################################################
# Flatten the annotations
################################################################################

# counts the number of positive examples each imname associated with
imexample_count = defaultdict(int)

flat_query_dict = {imname: [] for imname in imlist}
for imname in imlist:
    this_imcrop_names = imcrop_dict[imname]
    for imcrop_name in this_imcrop_names:
        if imcrop_name not in query_dict:
            continue
        this_descriptions = query_dict[imcrop_name]
        for description in this_descriptions:
            flat_query_dict[imname].append((description, 1))
            imexample_count[imname] += 1

# add false examples to the flat_query_dict
for i, imname in enumerate(imexample_count):
    this_imcrop_names = imcrop_dict[imname]
    
    j = 0
    while j < imexample_count[imname]:
        # choose random image index in imlist
        rand_img_idx = i
        while rand_img_idx == i:
            rand_img_idx = randint(0, len(imlist)-1)
        rand_imname = imlist[rand_img_idx]
        rand_imcrop_names = imcrop_dict[rand_imname]

        # choose random description by shuffling possible descriptions,
        # checking if each is in test set. If none are, must try choosing
        # new random index
        shuffle(rand_imcrop_names)
        rand_description = None
        for imcrop_name in rand_imcrop_names:
            if imcrop_name in query_dict:
                # always choose first caption, since most imcrops only have
                # one caption (and too bad for the other ones)
                rand_description = query_dict[imcrop_name][0]
                break

        # checks that we found a caption. If no good caption to choose from, start
        # loop over by choosing new random image
        if rand_description is not None:
            flat_query_dict[imname].append((rand_description, 0))
            j += 1

################################################################################
# Testing
################################################################################

# TODO: run through the model
# At this point, we have a dictionary called flat_query_dict which is
# key: image name, value: tuple of (description, label)
# Note that each image name has values that have real descriptions with label 1,
# and an equal number of "bad" descriptions with label 0.

correct_predictions = 0
total_predictions = 0

# Pre-allocate arrays, this does one img at a time
# imcrop_val = np.zeros((input_H, input_W, 3), dtype=np.float32)
# text_seq_val = np.zeros(T, dtype=np.int32)

imcrop_val = np.zeros((N, input_H, input_W, 3), dtype=np.float32)
text_seq_val = np.zeros((T, N), dtype=np.int32)

#print('text_seq_val before adding any vals:')
#print(text_seq_val)

num_im = len(imlist)
for n_im in range(num_im):
    print('testing image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    imsize = imsize_dict[imname]
    
    # Extract visual features from all proposals
    # Process image before testing
    im = skimage.io.imread(image_dir + imname)
    processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
    if processed_im.ndim == 2:
        processed_im = np.tile(processed_im[:, :, np.newaxis], (1, 1, 3))

    imcrop_val[...] = processed_im.astype(np.float32) - segmodel.vgg_net.channel_mean

    # Extract textual features from sentences
    for description, im_label in flat_query_dict[imname]:
        # Extract language feature
        print('description:')
        print(description)
        text_seq_val[:, 0] = text_processing.preprocess_sentence(description, vocab_dict, T)
        #print('so now i did something with test seq val and gonna print here')
        #print(text_seq_val)
        
        # TODO: ensure running through model correctly, this is the prediction step
        scores_val = sess.run(scores, feed_dict={
            text_seq_batch  : text_seq_val,
            imcrop_batch    : imcrop_val
        })
        #print('scores unmodified:')
        #print(scores_val)
        scores_val = np.squeeze(scores_val)
        #print('after squeezing:')
        #print(scores_val)

        # count if correct
        prediction = (scores_val > 0)
        correct_predictions += (prediction == im_label)
        total_predictions += 1

        print("%d correct_predictions out of %d" % (correct_predictions, total_predictions))

print('Final results on the whole test set')
result_str = 'recall = %f\n'.format(float(correct_predictions)/total_predictions)
print(result_str)
