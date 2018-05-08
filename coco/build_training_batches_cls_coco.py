from __future__ import absolute_import, division, print_function

import sys; sys.path.append('./coco')

import numpy as np
import os
import sys
import json
import skimage
import skimage.io
import skimage.transform

from coco.pycocotools.coco import COCO
from random import randint
from util import im_processing, text_processing
from util.io import load_referit_gt_mask as load_gt_mask

''' 
Sample execution:
- GloVe vocabulary: python coco/build_training_batches_cls_coco.py glove
- ReferIt vocabulary: python coco/build_training_batches_cls_coco.py referit
'''

################################################################################
# Parameters
################################################################################

image_dir = './coco/images/'
query_file = './coco/annotations/instances_train2017.json'
caption_file = './coco/annotations/captions_train2017.json'

# Saving directory
data_folder = './coco/data/train_batch_cls/'
data_prefix = 'coco_train_cls'

# Model Params
T = 20
N = 10 # number of items per batch
input_H = 224
input_W = 224

# num false samples per positive sample
F = 1

################################################################################
# Load annotations
################################################################################

coco = COCO(query_file)
coco_captions = COCO(caption_file)
imgid_list = coco.getImgIds()

################################################################################
# Load vocabulary
################################################################################

if sys.argv[1] == "glove":
    filename = './exp-referit/data/glove.6B.50d.txt'
    def loadGloVe(filename):
        vocab = []
        embd = []
        file = open(filename,'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
        print('Loaded GloVe!')
        file.close()
        return vocab,embd
    vocab,embd = loadGloVe(filename)
    vocab.append("<pad>")
    vocab_dict = dict()
    for i in range(len(vocab)): vocab_dict[vocab[i]] = i

else if sys.argv[1] == "referit":
    # use referit vocab file; extremely similar to top 8803 words in COCO vocab
    vocab_file = './exp-referit/data/vocabulary_referit.txt'
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

else:
    sys.exit("Invalid vocabulary chosen (argument 1).")

################################################################################
# Collect training samples
################################################################################

training_samples = []
false_samples = []
num_imcrop = len(imgid_list)

for n_imcrop in range(num_imcrop):
    if n_imcrop % 200 == 0: print('processing %d / %d' % (n_imcrop+1, num_imcrop))
    img_id = imgid_list[n_imcrop]

    # get the decriptions of the image
    caption_ids = coco_captions.getAnnIds(imgIds=img_id)
    captions = [x['caption'].strip() for x in coco_captions.loadAnns(caption_ids)]
    for description in captions:
        # Append F times to match num of false samples
        for i in range(F):
            training_samples.append((img_id, description, 1))

        # generate F false samples for each positive sample
        for i in range(F):
            # Choose random image that is not current image, get its descriptions,
            # and choose one at random
            false_idx = n_imcrop
            while false_idx == n_imcrop: false_idx = randint(0, num_imcrop-1)
            desc_ids = coco_captions.getAnnIds(imgid_list[false_idx])
            desc_idx = randint(0, len(desc_ids)-1)
            false_cap = coco_captions.loadAnns(desc_ids[desc_idx])[0]['caption'].strip()

            false_samples.append((img_id, false_cap, 0))

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
label_batch = np.zeros(N, dtype=np.bool)

if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
for n_batch in range(num_batch):
    print('saving batch %d / %d' % (n_batch+1, num_batch))
    batch_begin = n_batch * N
    batch_end = (n_batch+1) * N
    for n_sample in range(batch_begin, batch_end):
        img_id, description, label = shuffled_training_samples[n_sample]
        # load image and get host url for image
        imname = coco.loadImgs(img_id)[0]['coco_url']
        im = skimage.io.imread(imname)
        
        processed_im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
        if processed_im.ndim == 2:
            processed_im = processed_im[:, :, np.newaxis]

        text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)
        text_seq_batch[:, n_sample-batch_begin] = text_seq
        imcrop_batch[n_sample-batch_begin, ...] = processed_im
        label_batch[n_sample-batch_begin] = label

    np.savez(file=data_folder + data_prefix + '_' + str(n_batch) + '.npz',
         text_seq_batch=text_seq_batch,
         imcrop_batch=imcrop_batch,
         label_batch=label_batch)
