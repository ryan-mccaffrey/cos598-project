from __future__ import absolute_import, division, print_function

import sys; sys.path.append('./coco')
import os
import skimage.io
import numpy as np
import tensorflow as tf
import json
import timeit
import argparse

from random import randint
from matplotlib import pyplot as plt

from coco.pycocotools.coco import COCO
from models import vgg_net, lstm_net, processing_tools
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from util import im_processing, text_processing, eval_tools

################################################################################
# Parameters
################################################################################

image_dir = './coco/images/'
query_file = './coco/annotations/instances_val2017.json'
caption_file = './coco/annotations/captions_val2017.json'
# bbox_proposal_dir = './exp-referit/data/referit_edgeboxes_top100/'
# bbox_file = './exp-referit/data/referit_bbox.json'

# TODO: Change model name for coco
pretrained_model = './exp-referit/tfmodel/cls_coco_glove_45000.tfmodel'

# Model Params
T = 20
N = 100
lstm_dim = 1000
mlp_hidden_dims = 500

D_im = 1000
D_text = lstm_dim

neg_iou = 1e-6

################################################################################
# Load vocabulary
################################################################################

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

# Vocabulary-related network parameters
num_vocab = len(vocab)
embed_dim = len(embd[0])

################################################################################
# Evaluation method
################################################################################

def main(args):

    ################################################################################
    # Validate input arguments
    ################################################################################    
    assert not (args.concat and args.crops), "Cannot test concatenated labels on image crops."
    assert not (args.concat and (not args.multicrop)), "Cannot test concatenated labels on single image crop per batch."

    # TODO: remove if allowing crops again
    assert not args.crops, "Crops currently not supported"

    # Initialize GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ID    

    # print mode
    print()
    print("Crops - True | Full Images - False:", args.crops)
    print("All crops per batch - True | First crop per batch - False:", args.multicrop)
    print("Concatenated captions - True | Simple captions - False:", args.concat)
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

    coco = COCO(query_file)
	coco_captions = COCO(caption_file)
	imgid_list = coco.getImgIds()

    # query_dict = json.load(open(query_file))   # e.g.: "38685_1":["sky"]
    # bbox_dict = json.load(open(bbox_file))     # {"38685_1":[0,0,479,132]
    # imcrop_dict = json.load(open(imcrop_file)) # "7023.jpg":["7023_3","7023_7","7023_1","7023_6",
    #                                            #             "7023_5","7023_2","7023_4"]
    # imsize_dict = json.load(open(imsize_file)) # "7023.jpg":[480,360]
    # imlist = list({name.split('_', 1)[0] + '.jpg' for name in query_dict})
    # vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
    # imcrop_list = list(query_dict.keys())

    # # Object proposals
    # bbox_proposal_dict = {}
    # for imname in imlist:
    #     bboxes = np.loadtxt(bbox_proposal_dir + imname[:-4] + '.txt').astype(int).reshape((-1, 4))
    #     bbox_proposal_dict[imname] = bboxes

    ################################################################################
    # Load testing data
    ################################################################################

    count = 0
    testing_samples_pos = []
    testing_samples_neg = []
    num_imcrop = len(imgid_list)

    # if args.crops:
    #     # Gather a testing example per image crop.
    #     for imname in imlist:
    #         this_imcrop_names = imcrop_dict[imname]
    #         imsize = imsize_dict[imname]
    #         bbox_proposals = bbox_proposal_dict[imname]

    #         # for each ground-truth annotation, use gt boxes as positive examples
    #         # and proposal box with small iou as negative examples
    #         for imcrop_name in this_imcrop_names:
    #             if not imcrop_name in query_dict:
    #                 continue
    #             gt_bbox = np.array(bbox_dict[imcrop_name]).reshape((1, 4))
    #             IoUs = eval_tools.compute_bbox_iou(bbox_proposals, gt_bbox)
    #             pos_boxes = gt_bbox
    #             neg_boxes = bbox_proposals[IoUs <  neg_iou, :]

    #             # generate them per description; 
    #             # ensure equal number of positive and negative examples
    #             this_descriptions = query_dict[imcrop_name]
    #             for description in this_descriptions:
    #                 count += 1
    #                 # Positive testing samples
    #                 for n_pos in range(pos_boxes.shape[0]):
    #                     sample = (imname, imsize, pos_boxes[n_pos], description, 1)
    #                     testing_samples_pos.append(sample)
    #                 # Negative testing samples
    #                 for n_neg in range(min(neg_boxes.shape[0], pos_boxes.shape[0])):
    #                     sample = (imname, imsize, neg_boxes[n_neg], description, 0)
    #                     testing_samples_neg.append(sample)

    # else:
        
    # Gather a testing example per full image.
    for n_imcrop in range(num_imcrop):
        # image
        img_id = imgid_list[n_imcrop]

        # get the decriptions of the image
    	caption_ids = coco_captions.getAnnIds(imgIds=img_id)
    	captions = [x['caption'].strip() for x in coco_captions.loadAnns(caption_ids)]

        if args.concat:
            # append two positive captions; one with itself if only one present 
            # TODO: might just be better to choose 2 random pos captions either way?
            rand_idx1 = randint(0, len(captions))
            rand_idx2 = randint(0, len(captions))
            pos_desc = '{} and {}'.format(captions[rand_idx1], captions[rand_idx2])
            testing_samples_pos.append((img_id, pos_desc, 1))

            # OLD CONCAT CODE:
            # if len(captions) >= 2:
            #     pos_desc = captions[0] +  ' and ' + captions[1]
            #     testing_samples_pos.append((img_id, pos_desc, 1))
            # else: 
            #     pos_desc = captions[0] +  ' and ' + captions[0]
            #     testing_samples_pos.append((imname, pos_desc, 1))
               
            # form negative examples by choosing random image 
            # that is not the current image, get its descriptions,
            # and choose one at random.
            false_idx = n_imcrop
            while false_idx == n_imcrop: false_idx = randint(0, num_imcrop-1)
            desc_ids = coco_captions.getAnnIds(imgid_list[false_idx])
            desc_idx = randint(0, len(desc_ids)-1)
            neg_desc1 = coco_captions.loadAnns(desc_ids[desc_idx])[0]['caption'].strip()

            false_idx = n_imcrop
            while false_idx == n_imcrop: false_idx = randint(0, num_imcrop-1)
            desc_ids = coco_captions.getAnnIds(imgid_list[false_idx])
            desc_idx = randint(0, len(desc_ids)-1)
            neg_desc2 = coco_captions.loadAnns(desc_ids[desc_idx])[0]['caption'].strip()

            # negative example: append two negative captions
            neg_desc = neg_desc1 + ' and ' + neg_desc2                
            testing_samples_neg.append((img_id, neg_desc, 0))

            # negative example: append one negative and one positive example
            neg_desc = neg_desc1 + ' and ' + query_dict[imcrop_name][0]               
            testing_samples_neg.append((imname, neg_desc, 0))
            neg_desc = query_dict[imcrop_name][0] + ' and ' + neg_desc1          
            testing_samples_neg.append((img_id, neg_desc, 0))

        else:
            for caption in captions:
                # append one positive sample per description
                testing_samples_pos.append((img_id, caption, 1))
               
                # form one negative example by choosing random image 
                # that is not the current image, get its descriptions,
                # and choose one at random.
                false_idx = n_imcrop
                while false_idx == n_imcrop: false_idx = randint(0, num_imcrop-1)
                desc_ids = coco_captions.getAnnIds(imgid_list[false_idx])
            	desc_idx = randint(0, len(desc_ids)-1)
            	false_cap = coco_captions.loadAnns(desc_ids[desc_idx])[0]['caption'].strip()

                testing_samples_neg.append((img_id, false_cap, 0))


    # Combine samples
    print('#pos=', len(testing_samples_pos))
    print('#neg=', len(testing_samples_neg))
    print('Total img-captions pairs:', count)

    # TODO: Not exactly sure what your multicrop is testing here? Just removes the
    # positive examples from being tested? How is this useful?
    if args.multicrop:
        testing_samples = testing_samples_pos + testing_samples_neg
    else:
        testing_samples = testing_samples_neg

    # Merge and shuffle testing examples
    np.random.seed(3)
    perm_idx = np.random.permutation(len(testing_samples))
    shuffled_testing_samples = [testing_samples[n] for n in perm_idx]
    del testing_samples
    print('#total testing examples=', len(shuffled_testing_samples))

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

    correct_predictions = 0
    total_predictions = 0

    for n_batch in range(num_batch):
        print('batch %d / %d' % (n_batch+1, num_batch))
        batch_begin = n_batch * N
        batch_end = (n_batch+1) * N

        # load and preprocess first image per batch
        # TODO: Why max of (batch_begin-1, 0)? Doesn't that pick the image right before
        # the start of the batch?
        #first_imname = shuffled_testing_samples[max(batch_begin-1, 0)][0]
        first_img_id = shuffled_testing_samples[max(batch_begin-1, 0)][0]
        first_imname = coco.loadImgs(first_img_id)[0]['coco_url']
        first_im = skimage.io.imread(first_imname)
        first_imcrop = skimage.img_as_ubyte(skimage.transform.resize(first_im, [224, 224]))

        for n_sample in range(batch_begin, batch_end):
            # TODO: implement for crops in future, if decide crops are useful.
            if args.crops:
                imname, imsize, sample_bbox, description, label = shuffled_testing_samples[n_sample]
                im = skimage.io.imread(image_dir + imname)
                xmin, ymin, xmax, ymax = sample_bbox

                if len(np.shape(im)) == 3:
                    # grab bounding box from image
                    if args.multicrop:
                        imcrop = im[ymin:ymax+1, xmin:xmax+1, :]
                        imcrop = skimage.img_as_ubyte(skimage.transform.resize(imcrop, [224, 224]))
                    else:
                        imcrop = first_imcrop
                    text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)
                else:
                    # ignore grayscale images
                    continue

                # # Show Images
                # print(description, label)
                # fig=plt.figure()
                # a = fig.add_subplot(1, 2, 1)
                # plt.imshow(im)
                # a = fig.add_subplot(1, 2, 2)
                # plt.imshow(imcrop)
                # plt.show()
            else:
                img_id, description, label = shuffled_testing_samples[n_sample]
                imname = coco.loadImgs(img_id)[0]['coco_url']
                im = skimage.io.imread(imname)

                if len(np.shape(im)) == 3:
                    if args.multicrop:
                        imcrop = skimage.img_as_ubyte(skimage.transform.resize(im, [224, 224]))
                    else:
                        imcrop = first_imcrop
                    text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)
                else:
                    # ignore grayscale images
                    # imcrop = np.zeros((224, 224, 3), dtype=np.float32)
                    # text_seq = text_processing.preprocess_sentence(description, vocab_dict, T)
                    # label = 0
                    continue
                
            # Form batch
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
    result_str = 'recall = %f\n'.format(float(correct_predictions)/total_predictions)
    print(result_str)

'''
Sample execution: 
python exp-referit/exp_test_referit_cls_crop.py $GPU_ID --full --multiple --simple
--full: Full Images  vs --crops: Bounding Box Crops
--multiple: Mutliple images per batch vs --single: Singe image per batch
--concat: Concatenated labels vs --simple: Simple labels
'''
DESCRIPTION = """Performance evaluation suite for cls_crop model."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('GPU_ID', help='GPU_ID; if single-GPU, enter 0.')
    parser.add_argument('--crops', dest='crops', action='store_true')
    parser.add_argument('--full', dest='crops', action='store_false')
    parser.add_argument('--multiple', dest='multicrop', action='store_true')
    parser.add_argument('--single', dest='multicrop', action='store_false')
    parser.add_argument('--concat', dest='concat', action='store_true')
    parser.add_argument('--simple', dest='concat', action='store_false')
    args = parser.parse_args()
    main(args)

