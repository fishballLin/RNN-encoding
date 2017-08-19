import sys,os
import subprocess
'''
    This program will read all .txt file under current directory.
    Each line in <save_file_name> is lines in a .txt file separated by one <tab>.
'''

## usage messege
if len(sys.argv) != 2:
    print ('Usage : python3 preprocess.py <save_file_name>')
    sys.exit()

## get save path
save_file = sys.argv[1]

output_file = open(save_file,'w')

## walk through all file under current directory
for root, dirs, files in  os.walk('./pre_subtitle_no_TC'):
    for name in files:
        path = os.path.join(root,name)
        if os.path.isfile(path) and name.split('.')[-1]=='txt':
            print ('find %s'%path)
            
            ## read file
            with open(path,'r') as f:
                lines = [] 
                for line in f:
                    line = line.strip(' 　\n')
                    if line== '':
                        continue
                    lines.append(line)
                print('\t'.join(lines),file=output_file)

output_file.close()
            
            

