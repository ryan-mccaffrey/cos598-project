from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit

from models import vgg_net, lstm_net, processing_tools
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util import im_processing, text_processing, eval_tools

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
bbox_proposal_dir = './exp-referit/data/referit_edgeboxes_top100/'
query_file = './exp-referit/data/referit_query_test.json'
bbox_file = './exp-referit/data/referit_bbox.json'
imcrop_file = './exp-referit/data/referit_imcrop.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

pretrained_model = './exp-referit/tfmodel/referit_fc8_cls_crop_iter_0.tfmodel'

# Model Params
T = 20
N = 100
num_vocab = 8803
embed_dim = 1000
lstm_dim = 1000
mlp_hidden_dims = 500

D_im = 1000
D_text = lstm_dim

neg_iou = 1e-6

F = 1

################################################################################
# Evaluation network
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, N])
imcrop_batch = tf.placeholder(tf.float32, [N, 224, 224, 3])
lstm_top_batch = tf.placeholder(tf.float32, [N, D_text])
fc8_crop_batch = tf.placeholder(tf.float32, [N, D_im])

# Language feature (LSTM hidden state)
lstm_top = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)

# Local image feature
fc8_crop = vgg_net.vgg_fc8(imcrop_batch, 'vgg_local', apply_dropout=False)

# L2-normalize the features (except for spatial_batch)
# and concatenate them along axis 1 (feature dimension)
feat_all = tf.concat(axis=1, values=[tf.nn.l2_normalize(lstm_top_batch, 1),
                         tf.nn.l2_normalize(fc8_crop_batch, 1)])

# Outputs
# MLP Classifier over concatenate feature
with tf.variable_scope('classifier'):
    mlp_l1 = fc_relu('mlp_l1', feat_all, output_dim=mlp_hidden_dims)
    mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)
scores = mlp_l2

# Load pretrained model
snapshot_restorer = tf.train.Saver(None)
sess = tf.Session()
snapshot_restorer.restore(sess, pretrained_model)

################################################################################
# Load annotations and bounding box proposals
################################################################################

query_dict = json.load(open(query_file))   # e.g.: "38685_1":["sky"]                                          #             "7023_5","7023_2","7023_4"]
imsize_dict = json.load(open(imsize_file)) #"7023.jpg":[480,360]
imcrop_list = query_dict.keys()
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

################################################################################
# Load testing data
################################################################################

print("Total images:", len(imlist))
count = 0

testing_samples_pos = []
testing_samples_neg = []
num_imcrop = len(imcrop_list)

for n_imcrop in range(num_imcrop):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop+1, num_imcrop))
    imcrop_name = imcrop_list[n_imcrop]

    # Image
    imname = imcrop_name.split('_', 1)[0] + '.jpg'
    for description in query_dict[imcrop_name]:
        # append F times to match num of false samples
        for i in range(F):
            testing_samples_pos.append((imname, description, 1))

        # generate F false samples for each positive sample
        for i in range(F):
            # Choose random image that is not current image, get its descriptions,
            # and choose one at random
            false_idx = n_imcrop
            while false_idx == n_imcrop: false_idx = randint(0, num_imcrop-1)
            descriptions = query_dict[imcrop_list[false_idx]]
            desc_idx = randint(0, len(descriptions)-1)

            testing_samples_neg.append((imname, descriptions[desc_idx], 0))

# Combine samples
print('#pos=', len(testing_samples_pos))
print('#neg=', len(testing_samples_neg))
combined_samples = testing_samples_pos + testing_samples_neg

# Shuffle the testing instances
np.random.seed(3)
shuffle_idx = np.random.permutation(len(combined_samples))
shuffled_testing_samples = [combined_samples[n] for n in shuffle_idx]
print('total testing instance number: %d' % len(shuffled_testing_samples))

# Create training batches
num_batch = len(shuffled_testing_samples) // N
print('total batch number: %d' % num_batch)

################################################################################
# Testing
################################################################################

# Pre-allocate arrays
imcrop_val = np.zeros((N, 224, 224, 3), dtype=np.float32)
text_seq_val = np.zeros((T, N), dtype=np.int32)
lstm_top_val = np.zeros((N, D_text))
label_val = np.zeros((N, 1), dtype=np.bool)

print(num_batch * N, "batches collected.")
correct_predictions = 0
total_predictions = 0

for n_batch in range(num_batch):
    print('batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        imname, description, label = shuffled_testing_samples_pos[n_sample]
        im = skimage.io.imread(image_dir + imname)

        if len(np.shape(im)) == 3:
            imcrop = skimage.img_as_ubyte(skimage.transform.resize(imcrop, [224, 224]))
            text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)
        else:
            # ignore grayscale images to negative example
            continue
            # imcrop = np.zeros((224, 224, 3), dtype=np.float32)
            # text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)
            # label = 0
            
        idx = n_sample - batch_begin
        text_seq_val[:, idx] = text_seq
        imcrop_val[idx, ...] = imcrop - vgg_net.channel_mean
        label_val[idx] = label

    # Extract visual feature
    fc8_crop_val = sess.run(fc8_crop, feed_dict={imcrop_batch:imcrop_val})

    # Extract language feature
    lstm_top_val[...] = sess.run(lstm_top, feed_dict={text_seq_batch:text_seq_val})

    # Compute scores per proposal
    scores_val = sess.run(scores,
        feed_dict={
            lstm_top_batch:lstm_top_val,
            fc8_crop_batch:fc8_crop_val
        })
    scores_val = scores_val[:batch_end-batch_begin+1, ...].reshape(-1)
    
    # Evaluate on bounding labels
    for indx in range(len(scores_val)):
        correct_predictions += ((scores_val[indx] > 0) ==  label_val[indx])
        total_predictions += 1
        
    print("%d correct predictions out of %d" % (correct_predictions, total_predictions))
    print(correct_predictions/total_predictions)
        
print('Final results on the whole test set')
print(result_str)

