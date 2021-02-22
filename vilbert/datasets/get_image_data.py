# Get image height and width in a dictionary from the LMDB files

import lmdb
import os
import pickle
import json
import pdb
from tqdm import tqdm 

output_path = '/home/amitak/vilbert-multi-task/image_data.json'
lmdb_path = '/home/amitak/vilbert-multi-task/data/datasets/coco/features_100'
output_data = {}
pdb.set_trace()
for filename in ['trainval']: #, 'test':
    env = lmdb.open(os.path.join(lmdb_path, 'COCO_{}_resnext152_faster_rcnn_genome.lmdb'.format(filename)),\
        max_readers=1, readonly=True, lock=False, \
        readahead=False, meminit=False,)
    with env.begin(write=False) as txn:
        image_ids = pickle.loads(txn.get("keys".encode()))
    for image_id in tqdm(image_ids):
        with env.begin(write=False) as txn:
            item = pickle.loads(txn.get(image_id))
            if int(image_id) in output_data:
                pdb.set_trace()
                print("?")
            output_data[int(image_id)] = {}
            output_data[int(image_id)]['image_h'] = int(item["image_h"])
            output_data[int(image_id)]['image_w'] = int(item["image_w"])

wp = open(output_path, 'w')
json.dump(output_data, wp)

