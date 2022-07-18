import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import glob
from tqdm import tqdm

from voltaml.utils.torch_utils import select_device, time_sync
from voltaml.models.common import DetectMultiBackend
from voltaml.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

from torch.multiprocessing import Pool, Process, set_start_method

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.fn as fn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('test_data/uploads/'):
#     for filename in filenames:
#         # print(filename)
#         os.path.join(dirname, filename)
        
class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)
        
        input_path = '/workspace/voltaML/test_data/uploads/'
        
        self.img_list = glob.glob(input_path+'/*.png')
        # dummy_label = [0]*len(self.img_list)
        # df = pd.DataFrame({'data' : self.img_list, 'label' : dummy_label})
        # df.to_csv('dali.txt', header=False, index=False, sep=' ')
        
        # self.input = fn.readers.file(file_root='/workspace/')
        # self.decode = fn.decoders.image_random_crop(device="mixed", output_type=types.RGB)
        # self.resize = fn.resize(device = "gpu",
        #                          image_type = types.RGB,
        #                          resize_x=1280., resize_y=1280.)
        
        self.input = ops.FileReader(file_root=input_path)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu",
                                 image_type = types.RGB,
                                 # dtype = types.FLOAT,
                                 resize_x=1280., resize_y=1280.)

        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                           output_dtype = types.FLOAT,
                                            output_layout=types.NCHW,
                                            # mirror = 1
                                            image_type = types.RGB,
                                           # scale=1/255.
                                          )
                                            # mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            # std=[0.229 * 255,0.224 * 255,0.225 * 255])
                                            # mean = [255., 255., 255.])
                                            # std = [1., 1., 1.])
                        
        self.cmn_2 = ops.CropMirrorNormalize(device = "gpu",
                                           scale=1/255.
                                          )
        
    def define_graph(self):
        images, labels = self.input(name="Reader")
        images = self.decode(images)
        images = self.resize(images)
        images = self.cmn(images)
        output = self.cmn_2(images)
        return output

class DALICustomIterator(DALIGenericIterator):
    def __init__(self, pipelines, output_map, size, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False):
        super(DALICustomIterator, self).__init__(pipelines, output_map, size, auto_reset, fill_last_batch, dynamic_shape, last_batch_padded)

    def __len__(self):
        return int(self._size / self.batch_size) + 1

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch
        feed = super().__next__()
        data = feed[0]['data']
        return data

def DALIDataLoader(batch_size):
    num_gpus = 1
    pipes = [DALIPipeline(batch_size=batch_size,
                          num_threads=128, device_id=device_id) for device_id in range(num_gpus)]
    pipes[0].build()
    dali_iter = DALICustomIterator(pipes, ['data'], pipes[0].epoch_size("Reader"))
    return dali_iter


def dali_infer(model_path, batch_size):
    start = time_sync()
    data_loader = DALIDataLoader(batch_size=batch_size )
        
    model_dir = [model_path]
    device = select_device('0')
    model = DetectMultiBackend(model_dir, device=device)
    
    dt = [0.0, 0.0, 0.0]
    total_pred = 0
    for img_batch in tqdm(data_loader):
        # print(img_batch.shape)
        # if img_batch.shape[0] < batch_size:
        #     continue

        t2 = time_sync()
        # Inference
        try:
            pred = model(img_batch, augment=False, visualize=False)
        except:
            continue
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        conf_thres, iou_thres, classes, agnostic_nms, max_det= 0.4, 0.1, None, False, 1000
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        # print(pred)

        for det in pred:
            if det.shape[0] > 0:
                total_pred += 1
    dt[0] = time_sync() - start

    seen = batch_size*len(data_loader)
    print('Total preds :', total_pred)


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    tt_s = tuple(x for x in dt)  # total time in sec
    tt_ms = tuple(x * 1E3 for x in dt)  # total time in ms
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape ' % t)
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape ' % tt_ms)
    print(f'Speed: %.1fs pre-process, %.1fs inference, %.1fs NMS per image at shape ' % tt_s)    
    return t, tt_s, tt_ms, total_pred, seen

if __name__ == '__main__':

    set_start_method('spawn')
    dali_infer(model_path='test_models/best_30k_data.pt', batch_size=1)