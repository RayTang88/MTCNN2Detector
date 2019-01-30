import numpy as np
import os

from torch.utils.data import Dataset
from PIL import Image



class Mydataset():
    def __init__(self, img_path):

        self.img_path = img_path
        self.data()A

    def data(self):

        self.dataset = []
        self.dataset.extend(open(os.path.join(self.img_path, 'positive.txt'), 'r'))
        self.dataset.extend(open(os.path.join(self.img_path, 'negative.txt'), 'r'))
        self.dataset.extend(open(os.path.join(self.img_path, 'part.txt'), 'r'))


        for lines in self.dataset:



            line = lines.strip().split()
            filename = line[0]
            offx1 = float(line[1])
            offy1 = float(line[2])
            offx2 = float(line[3])
            offy2 = float(line[4])
            cond = float(line[5])
            lable = np.array([offx1, offy1, offx2, offy2])
            img_data = np.array(Image.open(os.path.join(self.img_path, filename)))
            print(cond, lable)




        return img_data, cond,lable


if __name__ == '__main__':
    pnet_img_path = '/home/ray/datasets/Mtcnn/test/12'
    mydataset = Mydataset(pnet_img_path)
