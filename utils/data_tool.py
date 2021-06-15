import os
from tqdm import tqdm

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
    
