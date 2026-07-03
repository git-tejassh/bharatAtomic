import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
from pathlib import Path 

path = '/Users/tjsss/Desktop/bharatAtomic/semPhase2/main_folder/models/LBMS_SAM/emps_dataset/emps-DatasetNinja (2)/ds/' 

class DataLoading():
    def __init__ (self, path: str,
                  val_dir_exists: bool = False,
                  test_dir_exists: bool = False
    ):
        
        super.__init__()
        self.path = path
        self.val_dir_exists = val_dir_exists
        self.test_dir_exists = test_dir_exists


    def split_images(self,):
        ## TO BE DONE ONCE
        val_dir = path + 'val_dir'
        test_dir = path + 'test_dir'
        if self.val_dir_exists == False:
            Path(val_dir).mkdir(parents = True, exist_ok=True)
            self.val_dir_exists = True
        else:
            print("Folder already exists")
        if self.test_dir_exists == False:
            Path(test_dir).mkdir(parents=True, exist_ok=True)
            self.test_dir_exists = True
        else:
            print("Folder already exists")
            
            


    def load_images(self,):
        images_dir = os.path.join(path, 'img')
        ann_dir = os.path.join(path, 'ann')

        images = np.array(os.listdir(images_dir), dtype = object)
        ann = np.array(os.listdir(ann_dir), dtype = object)

        print(f"Number of images: {len(images)}")
        print(f"Number of annotations: {len(ann)}")

        
