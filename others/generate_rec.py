import os
import subprocess
import pandas as pd
from ..data.config import cfg

def Generate_Record(dataset):

    data_type = dataset.data_type

    records = []
    for index, i in  enumerate(['img1','img2','label']):
        df = pd.DataFrame(dataset.dirs)
        df[index].to_csv(cfg.record_prefix + '{}_{}_{}.lst'.format(dataset.name(),data_type,i),sep='\t',header=False)
        args = ['python','im2rec.py',cfg.record_prefix +'{}_{}_{}'.format(dataset.name(),data_type,i) , \
                '--root','','--resize','0','--quality','0','--num_thread','1','--encoding', '.png']
        subprocess.call(args)

def get_imageRecord(dataset,batchsize,prefetch_buffer):

    data_type = dataset.data_type

    records = []
    for index,i in  enumerate(['img1','img2','label']):

        if not os.path.exists(cfg.record_prefix+'{}_{}_{}.rec'.format(dataset.name(),data_type,i)):

            df = pd.DataFrame(dataset.dirs)
            df[index].to_csv(cfg.record_prefix + '{}_{}_{}.lst'.format(dataset.name(),data_type,i),sep='\t',header=False)
            args = ['python','im2rec.py',cfg.record_prefix +'{}_{}_{}'.format(dataset.name(),data_type,i) , \
                    '--root','','--resize','0','--quality','0','--num_thread','1','--encoding', '.png']
            subprocess.call(args)

        records.append(mx.io.ImageRecordIter(
            path_imgrec = cfg.record_prefix+'{}_{}_{}.rec'.format(dataset.name(),data_type,i),
            data_shape = dataset.shapes(),
            batch_size = batchsize,
            preprocess_threads = 3,
            prefetch_buffer = prefetch_buffer,
            shuffle = False))
    return records