import os
import tqdm
sdfpath = './SDFGen/bin/SDFGen'


root = '/home/panmingjie/gsnet_data/data_obj'

with open('filtered_obj_list.txt', 'r') as f:
    obj_names = f.read().splitlines()
    


for obj_name in tqdm.tqdm(obj_names, total=len(obj_names)):
    obj_path = os.path.join(root, obj_name)
    print("Processing %s" % obj_path)
    os.system('%s %s %d %d' % (sdfpath, obj_path, 100, 5))
