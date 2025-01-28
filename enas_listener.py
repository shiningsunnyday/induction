import torch
import torch.multiprocessing as mp
import numpy as np
import fcntl
import argparse
import setproctitle
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append('/home/msun415/induction/dagnn/dvae/software/enas/src/cifar10')    
from evaluation import *
eva = Eval_NN()


def lock(f):
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        return False
    return True


def main(proc_id, filename, output_filename):  
    while not os.path.exists(filename): # temporary
        print(os.path.abspath(filename), "not exist")
        time.sleep(1)
    while(True):
        selected_arc = None
        with open(filename, 'r') as f:
            editable = lock(f)
            if editable:
                lines = f.readlines()
                num_samples = len(lines)
                new_lines = []
                for idx, line in enumerate(lines):
                    splitted_line = line.strip().split()
                    if len(splitted_line) == 1 and (selected_arc is None):
                        selected_arc = (idx, splitted_line[0])
                        new_line = "{} {}\n".format(splitted_line[0], "working")
                    else:
                        new_line = "{}\n".format(" ".join(splitted_line))
                    new_lines.append(new_line)
                with open(filename, 'w') as fw:
                    for _new_line in new_lines:
                        fw.write(_new_line)
                fcntl.flock(f, fcntl.LOCK_UN)
        if selected_arc is None:
            continue
        
        print("====Working for sample {}/{}====".format(selected_arc[0], num_samples))
        result = eva.eval(selected_arc[1].replace(',', ' '))


        while(True):
            with open(output_filename, 'a') as f:
                editable = lock(f)
                if editable:
                    f.write("{} {} {}\n".format(selected_arc[0], selected_arc[1], result))
                    fcntl.flock(f, fcntl.LOCK_UN)
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='enas listener')
    parser.add_argument('--proc_id', type=int, default=1, help="process id")
    parser.add_argument('--filename', type=str, default="enas_in.txt", help="file name to lister")
    parser.add_argument('--output_filename', type=str, default="enas_out.txt", help="file name to output")
    args = parser.parse_args()
    setproctitle.setproctitle("enas_listener")
    main(args.proc_id, args.filename, args.output_filename)
