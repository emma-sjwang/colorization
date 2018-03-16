# sjwang
# created on 2018/01/27

from produce_data import produce_data
import os

def generate(q, phase, data_path_root, filelist):
    '''
    generator data not stop
    :param q:
    :param phase:
    :param data_path_root:
    :param filelist:
    :return:
    '''
    index = 0

    while 1:
        produce_data(q, phase, data_path_root, filelist)
        index += 1


def produce(q, phase, data_path_root):
    '''
    generator data
    :param q:
    :param phase:
    :param val_num:
    :return:
    '''

    filelist = "../data/%s_data_list.txt" % phase
    if not os.path.exists(filelist):
        print("ERROR: %s is not exits! in produce_only.py." % filelist)
        exit(-1)
    generate(q, phase, data_path_root, filelist)

