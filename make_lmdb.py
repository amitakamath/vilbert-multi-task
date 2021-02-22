# Convert HDF5 to LMDB

import lmdb
import pickle
import json
import pdb
import os
import numpy as np
import h5py
from tqdm import tqdm

MAP_SIZE = 1099511627776

image_data = json.load(open('image_data.json', 'r'))
env = lmdb.open('trainval.lmdb', map_size=MAP_SIZE)
seen_image_ids = {}

with env.begin(write=True) as txn:
    for split in 'train', 'val':
        #count = 0
        f = h5py.File('/home/amitak/{}2014.hdf5'.format(split), 'r')
        for k, v in tqdm(f.items()):
            #pdb.set_trace()
            image_id = int(k.split('_')[-1])
            #count += 1
            #if count > 10:
            #    break
            # This doesn't happen:
            #if image_id in seen_image_ids:
            #    pdb.set_trace()
            #    print("?")
            seen_image_ids[str(image_id).encode()] = 1
            item = {}
            item['image_id'] = image_id
            item['image_h'] = image_data[str(image_id)]['image_h']
            item['image_w'] = image_data[str(image_id)]['image_w']
            item['num_boxes'] = 100
            current_num_boxes = len(v['boxes'][()])
            #print(current_num_boxes)
            item['boxes'] = np.pad(v['boxes'][()], [(0, 100-current_num_boxes), (0, 0)], mode='constant')
            item['features'] = np.pad(v['feats'][()], [(0, 100-current_num_boxes), (0, 0)], mode='constant')
            txn.put(str(image_id).encode(), pickle.dumps(item))
    id_list = list(seen_image_ids.keys())
    txn.put("keys".encode(), pickle.dumps(id_list))


