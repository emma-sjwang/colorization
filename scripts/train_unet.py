# sjwang
# created on 2018/01/27

from unet import consume
from produce_only import produce
from multiprocessing import Process, Queue
import sys

if __name__ == '__main__':

    # default use all Gpus, you can change to which id you want to use
    GPU_ID = '0,1,2,3'
    GPU_ID_list = []
    # GPU_ID = sys.argv[1]

    for i_gpu in GPU_ID.split(","):
        GPU_ID_list.append(int(i_gpu))

    # training data root path
    data_path_root = "../data/testSetPlaces205_resize/data/"
    q_train = Queue(2000)
    q_val = Queue(2)
    p = []
    p.append(Process(target = produce, args = (q_train,'train',data_path_root, )))
    p.append(Process(target = consume, args = (q_train,q_val, GPU_ID_list, data_path_root, )))


    for i in range(len(p)):
        print "start P %d" % i
        p[i].start()


    for i in range(len(p)):
        p[i].join()

    print "Train Done!!!  HAPPY NEW YEAR!!!"
    exit()
