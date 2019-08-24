import h5py
import os
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
import csv
import base64
import pickle
import lmdb # install lmdb by "pip install lmdb"

csv.field_size_limit(sys.maxsize)

count = 0
infiles = []

path = '/srv/share2/jlu347/bottom-up-attention/feature/coco/trainval/'
infiles.append(path + 'karpathy_train_resnet101_faster_rcnn_genome.tsv.0')
infiles.append(path + 'karpathy_train_resnet101_faster_rcnn_genome.tsv.1')
infiles.append(path + 'karpathy_val_resnet101_faster_rcnn_genome.tsv')
infiles.append(path + 'karpathy_test_resnet101_faster_rcnn_genome.tsv')

save_path = os.path.join('coco_trainval_resnet101_faster_rcnn_genome.lmdb')
env = lmdb.open(save_path, map_size=1099511627776)

id_list = []
with env.begin(write=True) as txn:
    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                img_id = str(item['image_id']).encode()
                id_list.append(img_id)
                txn.put(img_id, pickle.dumps(item))
                if count % 1000 == 0:
                    print(count)
                count += 1
    txn.put('keys'.encode(), pickle.dumps(id_list))

print(count)