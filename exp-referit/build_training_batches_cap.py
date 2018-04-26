
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import json
import skimage
import skimage.io
import skimage.transform

from random import randint
from util import im_processing, text_processing
from util.io import load_referit_gt_mask as load_gt_mask

################################################################################
# Parameters
################################################################################

image_dir = './exp-referit/referit-dataset/images/'
query_file = './exp-referit/data/referit_query_trainval.json'
imsize_file = './exp-referit/data/referit_imsize.json'
vocab_file = './exp-referit/data/vocabulary_referit.txt'

# Saving directory
data_folder = './exp-referit/data/train_batch_cls/'
data_prefix = 'referit_train_cls'

# Model Params
T = 20
N = 10 # number of items per batch
input_H = 224
input_W = 224

# num false samples per positive sample
F = 5

################################################################################
# Load annotations
################################################################################

query_dict = json.load(open(query_file))
imsize_dict = json.load(open(imsize_file))
imcrop_list = query_dict.keys()
vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

################################################################################
# Collect training samples
################################################################################

training_samples = []
false_samples = []
num_imcrop = len(imcrop_list)

for n_imcrop in range(num_imcrop):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop+1, num_imcrop))
    imcrop_name = imcrop_list[n_imcrop]

    # Image and mask
    imname = imcrop_name.split('_', 1)[0] + '.jpg'
    # mask_name = imcrop_name + '.mat'
    for description in query_dict[imcrop_name]:
        # Append F times to match num of false samples
        for i in range(F):
            training_samples.append((imname, description, 1))

        # generate F false samples for each positive sample
        for i in range(F):
            # Choose random image that is not current image, get its descriptions,
            # and choose one at random
            false_idx = n_imcrop
            while false_idx == n_imcrop: false_idx = randint(0, num_imcrop-1)
            descriptions = query_dict[imcrop_list[false_idx]]
            desc_idx = randint(0, len(descriptions)-1)

            false_samples.append((imname, descriptions[desc_idx], 0))

combined_samples = training_samples + false_samples

# Shuffle the training instances
np.random.seed(3)
shuffle_idx = np.random.permutation(len(combined_samples))
shuffled_training_samples = [combined_samples[n] for n in shuffle_idx]
print('total training instance number: %d' % len(shuffled_training_samples))

# Create training batches
num_batch = len(shuffled_training_samples) // N
print('total batch number: %d' % num_batch)

################################################################################
# Save training samples to disk
################################################################################

text_seq_batch = np.zeros((T, N), dtype=np.int32)
imcrop_batch = np.zeros((N, input_H, input_W, 3), dtype=np.uint8)
# label_coarse_batch = np.zeros((N, featmap_H, featmap_W, 1), dtype=np.bool)
# label_fine_batch = np.zeros((N, input_H, input_W, 1), dtype=np.bool)
pos_exnum_batch = np.zeros(N, dtype=np.bool)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for n_batch in range(num_batch):
    print('saving batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        imname, description, pos_example = shuffled_training_samples[n_sample]
        # imname, mask_name, description = shuffled_training_samples[n_sample]
        im = skimage.io.imread(image_dir + imname)
        # mask = load_gt_mask(mask_dir + mask_name).astype(np.float32)

        processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
        if processed_im.ndim == 2:
            processed_im = processed_im[:, :, np.newaxis]
        # processed_mask = im_processing.resize_and_pad(mask, input_H, input_W)
        # subsampled_mask = skimage.transform.downscale_local_mean(processed_mask, (32, 32))

        # labels_fine = (processed_mask > 0)
        # labels_coarse = (subsampled_mask > 0)

        text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)

        text_seq_batch[:, n_sample-batch_begin] = text_seq
        imcrop_batch[n_sample-batch_begin, ...] = processed_im
        # label_coarse_batch[n_sample-batch_begin, ...] = labels_coarse[:, :, np.newaxis]
        # label_fine_batch[n_sample-batch_begin, ...] = labels_fine[:, :, np.newaxis]
        pos_exnum_batch[n_sample-batch_begin] = pos_example

    np.savez(file=data_folder + data_prefix + '_' + str(n_batch) + '.npz',
         text_seq_batch=text_seq_batch,
         imcrop_batch=imcrop_batch,
         pos_exnum_batch=pos_exnum_batch)
