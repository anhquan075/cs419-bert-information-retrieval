import sys 
import json
import os
import random

import numpy as np

query_json_path = sys.argv[1]
output_fold = sys.argv[2]
os.makedirs(output_fold, exist_ok=True)
k_fold = 5

with open(query_json_path, 'r') as fo:
    query_lst = json.load(fo)

query_id_lst = [q['query_id'] for q in query_lst]
print(query_id_lst)

for idx in range(1, k_fold + 1):
    if idx == 1:
        test_idx = random.sample(query_id_lst, k = int(len(query_id_lst) * 0.2))
        substract_idx = list(set(query_id_lst) - set(test_idx))  
    else:
        test_idx = random.sample(substract_idx, k = int(len(query_id_lst) * 0.2))
        substract_idx = list(set(substract_idx) - set(test_idx))  
    
    train_idx = list(set(query_id_lst) - set(test_idx))  
    
    print('Test index', sorted(test_idx))
    print('Train index', sorted(train_idx))
        
    np.save(os.path.join(output_fold, f'train_index_fold{idx}.npy'), train_idx)
    np.save(os.path.join(output_fold, f'test_index_fold{idx}.npy'), test_idx)