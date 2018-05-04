# import sys
from collections import defaultdict
from pycocotools.coco import COCO
import operator
import json
import re

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
VOCAB_SIZE = 8803

default_words = ['<pad>', '<go>', '<eos>', '<unk>']

train_file='./coco/annotations/captions_train2017.json'
test_file='./coco/annotations/captions_val2017.json'
vocab_file = './coco/data/vocabulary_coco.txt'

vocab_to_freq = defaultdict(int)

# initialize COCO api for instance annotations
train_annotations = json.load(open(train_file))['annotations']
test_annotations = json.load(open(test_file))['annotations']

for annotation in test_annotations:
    caption = annotation['caption']
    words = SENTENCE_SPLIT_REGEX.split(caption.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    if words[-1] == '.':
        words = words[:-1]
    for word in words:
        vocab_to_freq[word] += 1
print('Completed populating test annotations...')

for annotation in train_annotations:
    caption = annotation['caption']
    words = SENTENCE_SPLIT_REGEX.split(caption.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    if words[-1] == '.':
        words = words[:-1]
    for word in words:
        vocab_to_freq[word] += 1
print('Completed populating train annotations...')

with open(vocab_file, 'w+') as f:
    # first write default vocab
    for word in default_words:
        f.write(word + '\n')

    num_words = VOCAB_SIZE - len(default_words)
    count = 0
    # sort words descending by value, ascending lexicographically
    for word, _ in sorted(vocab_to_freq.items(), key=lambda x: (-x[1], x[0])):
        count += 1
        f.write(word + '\n')
        if count >= num_words:
            break
