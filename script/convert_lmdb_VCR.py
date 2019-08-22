import h5py
import os
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import csv
import base64
import json_lines
import lmdb # install lmdb by "pip install lmdb"
import pickle
csv.field_size_limit(sys.maxsize)

def converId(img_id):

    img_id = img_id.split('-')
    if 'train' in img_id[0]:
        new_id = int(img_id[1])
    elif 'val' in img_id[0]:
        new_id = int(img_id[1]) + 1000000        
    elif 'test' in img_id[0]:
        new_id = int(img_id[1]) + 2000000    
    else:
        pdb.set_trace()

    return new_id

image_path = {}
path2id = {}
metadata_path = {}
with open('train.jsonl', 'rb') as f: # opening file in binary(rb) mode    
    for item in json_lines.reader(f):
        if item['img_fn'] not in image_path:
            image_path[item['img_fn']] = 1
            img_id = item['img_id']
            metadata_path[item['metadata_fn']] = 1
            path2id[item['img_fn']] = converId(item['img_id'])

with open('val.jsonl', 'rb') as f: # opening file in binary(rb) mode    
    for item in json_lines.reader(f):
        if item['img_fn'] not in image_path:
            image_path[item['img_fn']] = 1
            path2id[item['img_fn']] = converId(item['img_id'])
            metadata_path[item['metadata_fn']] = 1

with open('test.jsonl', 'rb') as f: # opening file in binary(rb) mode    
    for item in json_lines.reader(f):
        if item['img_fn'] not in image_path:
            image_path[item['img_fn']] = 1
            path2id[item['img_fn']] = converId(item['img_id'])
            metadata_path[item['metadata_fn']] = 1

count = 0
num_file = 4
name = '/srv/share2/jlu347/bottom-up-attention/feature/VCR/VCR_resnet101_faster_rcnn_genome.tsv.%d'
infiles = [name % i for i in range(num_file)]

id_list = []
save_path = os.path.join('VCR_resnet101_faster_rcnn_genome.lmdb')
env = lmdb.open(save_path, map_size=1099511627776)
with env.begin(write=True) as txn:

    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                img_id = str(path2id[item['image_id']]).encode()
                id_list.append(img_id)
                # txn.put(img_id, pickle.dumps(item))
                if count % 1000 == 0:
                    print(count)
                count += 1
    txn.put('keys'.encode(), pickle.dumps(id_list))

print(count)
json.dump(path2id, open('VCR_ImagePath2Id.json', 'w'))
