from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import deconv_layer as deconv
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from models import vgg_net, lstm_net
from models.processing_tools import *

def text_objseg_cls(text_seq_batch, imcrop_batch, num_vocab, embed_dim,
            lstm_dim, mlp_hidden_dims, vgg_dropout, mlp_dropout):

    # Language feature (LSTM hidden state)
    feat_lang = lstm_net.lstm_net(text_seq_batch, num_vocab, embed_dim, lstm_dim)
    print(feat_lang.get_shape())

    # Local image feature
    feat_vis = vgg_net.vgg_fc8(imcrop_batch, 'vgg_local', apply_dropout=vgg_dropout)
    print(feat_vis.get_shape())

    # L2-normalize the features (except for spatial_batch)
    # and concatenate them
    feat_all = tf.concat(axis=1, values=[tf.nn.l2_normalize(feat_lang, 1),
                                         tf.nn.l2_normalize(feat_vis, 1),])
    print(feat_all.get_shape())
    
    # MLP Classifier over concatenate feature
    with tf.variable_scope('classifier'):
        mlp_l1 = fc_relu('mlp_l1', feat_all, output_dim=mlp_hidden_dims)
        if mlp_dropout: mlp_l1 = drop(mlp_l1, 0.5)
        mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)

    return mlp_l2

def text_objseg_cls_glove(text_seq_batch, imcrop_batch, num_vocab, embed_dim,
                          lstm_dim, mlp_hidden_dims, vgg_dropout, mlp_dropout, embedding):

    # Language feature (LSTM hidden state)
    feat_lang = lstm_net.lstm_net_glove(text_seq_batch, embedding, lstm_dim)
    print(feat_lang.get_shape())

    # Local image feature
    feat_vis = vgg_net.vgg_fc8(imcrop_batch, 'vgg_local', apply_dropout=vgg_dropout)
    print(feat_vis.get_shape())

    # L2-normalize the features (except for spatial_batch)
    # and concatenate them
    feat_all = tf.concat(axis=1, values=[tf.nn.l2_normalize(feat_lang, 1),
                                         tf.nn.l2_normalize(feat_vis, 1),])
    print(feat_all.get_shape())
    
    # MLP Classifier over concatenate feature
    with tf.variable_scope('classifier'):
        mlp_l1 = fc_relu('mlp_l1', feat_all, output_dim=mlp_hidden_dims)
        if mlp_dropout: mlp_l1 = drop(mlp_l1, 0.5)
        mlp_l2 = fc('mlp_l2', mlp_l1, output_dim=1)

    return mlp_l2
    