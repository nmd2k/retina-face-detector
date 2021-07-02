import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_exp_dir(root_path='.'):
    for idx in range(1, 1000, 1):
        if not os.path.exists(os.path.join(root_path, f'exp{idx}')):
            save_dir = os.path.join(root_path, f'exp{idx}')
            break
    os.makedirs(save_dir)

    return save_dir

def exact_label(path, des):
    list_file = os.listdir(os.path.join(path, 'images'))

    # check folder exist
    for file in list_file:
        dir = os.path.join(path, 'labels', file)
        if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(path, 'labels/label.txt')) as reader:
        lines = reader.readlines()
        pbar  = tqdm(range(len(lines)))

        temp_file  = ''
        temp_lines = []
        
        for idx in pbar:
            new_line = lines[idx].split()

            if new_line[0] == '#':
                if temp_file != '':
                    with open(os.path.join(path, 'labels', temp_file), 'w') as writer:
                        writer.write(''.join(temp_lines))

                # go to new file and reset string writer
                temp_file  = new_line[1][:-3] + 'txt'
                temp_lines = []
            
            else:
                temp_lines.append(lines[idx])

def analysis_bb(root_path, is_train=True, out='research'):
    if is_train:
        path = os.path.join(root_path, 'train/labels')
    else:
        path = os.path.join(root_path, 'val/labels')

    dirname = os.listdir(path)

    statis_dict = {}

    for dir in dirname:
        subdir  = os.listdir(os.path.join(path, dir))
        pbar = tqdm(range(len(subdir)))
        pbar.set_description(dir, refresh=False)
        for idx in pbar:
            f = open(os.path.join(path, dir, subdir[idx]), 'r')
            lines = f.readlines()

            for jdx, line in enumerate(lines):
                line = line.strip().split()
                line = [float(x) for x in line]

                # bbox
                x1 = line[0]               # x1
                y1 = line[1]               # y1
                x2 = line[0] + line[2]     # x2
                y2 = line[1] + line[3]     # y2

                # csv
                try:
                    ap = np.around((x2-x1)/(y2-y1), 3)
                except:
                    # divide by 0
                    ap = -1
                statis_dict[subdir[idx][:-4]+f':{jdx}'] = [(x2-x1), (y2-y1), ap]
            
    
    df = pd.DataFrame.from_dict(statis_dict, orient='index', columns=['width', 'height', 'aspect_ratio'])
    df.to_csv(out+'.csv')
    
