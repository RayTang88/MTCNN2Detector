
import torch
import numpy as np
from torchvision import transforms
from torch import nn
import os
from PIL import Image, ImageDraw
from net import PNet, RNet, ONet
from utils import NMS, Re2Sq

class Detect():
    def __init__(self, pnet_para_path, rnet_para_path, onet_para_path, iscuda):
        self.iscuda = True
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        if self.iscuda == True:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        self.pnet.load_state_dict(torch.load(pnet_para_path))
        self.rnet.load_state_dict(torch.load(rnet_para_path))
        self.onet.load_state_dict(torch.load(onet_para_path))

        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()



        self.img_tensor = transforms.Compose([transforms.ToTensor()])



    def detect(self, image):

        # img_data = torch.Tensor(np.array(image)/255)
        #
        # p_img_data = torch.unsqueeze(img_data, 0)
        #
        # pu_img_data = p_img_data.permute(0, 3, 1, 2)
        #
        # if self.iscuda == True:
        #
        #     pu_img_data = pu_img_data.cuda()


        pnet_box = self._pnet(image, stride = 2, side_len = 12)

        rnet_box = self._rnet(image, pnet_box, side_len = 24)

        # onet_box = self._onet(image, rnet_box, side_len = 48)

        return rnet_box

    def _pnet(self, image, stride, side_len):
        # boxes = []
        # img=image
        # w,h=img.size
        # min_side_len = min(w, h)
        #
        #
        # scale = 1
        # while min_side_len > 24:
        #     img_data = self.img_tensor(img)
        #     if self.iscuda:
        #         img_data = img_data.cuda()
        #     img_data.unsqueeze_(0)
        #     _cls, _offset = self.pnet(img_data)
        #
        #
        #     cls, offset = _cls[0][0].cpu().data, _offset[0].cpu().data
        #
        #
        #     index_ = np.where(cls > 0.6)
        #     index = np.stack([index_[0], index_[1]], axis=1)
        #
        #
        #     offset = offset.numpy()
        #
        #     # print(index[0])
        #     # print(index[1])
        #     # print('np.stack',np.stack([index[0], index[1]], axis=1))
        #     # print(cls)
        #
        #     # index = torch.nonzero(torch.gt(cls, 0.6))
        #
        #     for _index in index:
        #
        #         _x1 = (_index[1] * stride) / scale
        #
        #         _y1 = (_index[0] * stride) / scale
        #         _x2 = (_index[1] * stride + side_len) / scale
        #         _y2 = (_index[0] * stride + side_len) / scale
        #
        #         _x1 = float(_x1)
        #
        #         _y1 = float(_y1)
        #         _x2 = float(_x2)
        #         _y2 = float(_y2)
        #
        #         ow = float(_x2 - _x1)
        #         oh = float(_y2 - _y1)
        #
        #         _offset = offset[:, _index[0], _index[1]]
        #
        #         x1 = _x1 + ow * _offset[0]
        #         y1 = _y1 + oh * _offset[1]
        #         x2 = _x2 + ow * _offset[2]
        #         y2 = _y2 + oh * _offset[3]
        #         boxes.append(np.array([x1,y1,x2,y2,cls[_index[0], _index[1]]]))
        #
        #         # boxes.append([self._box(idx, offset, cls[idx[0], idx[1]], scale)])
        #
        #     scale *= 0.7
        #
        #     _w = int(w * scale)
        #     _h = int(h * scale)
        #     img = img.resize((_w, _h))
        #     min_side_len = min(_w, _h)
        #
        #     count += 1
        #     print(np.array(boxes), np.array(boxes).shape)
        #     # boxes_ = np.squeeze(boxes, axis=1)
        #
        # return NMS(np.array(boxes), 0.3, False)



        pnetbox_ = []

        img = image
        w, h = img.size

        len = min(h, w)
        scale = 1

        while len > 24:

            # print(img.size)
            # print(scale)
            img_data = torch.Tensor(np.array(img) / 255-0.5)

            p_img_data = torch.unsqueeze(img_data, 0)

            pu_img_data = p_img_data.permute(0, 3, 1, 2)
            # img_data = self.img_tensor(img)

            # pu_img_data = self.img_tensor(img)
            # img_data.unsqueeze_(0)
            # pu_img_data.unsqueeze_(0)


            if self.iscuda == True:
                # img_data = img_data.cuda()
                pu_img_data = pu_img_data.cuda()
            # _cls, _offset = self.pnet(img_data)

            cond, offset = self.pnet(pu_img_data)

            # cls, offset = _cls[0][0].cpu().data, _offset[0].cpu().data
            cond = cond[0][0].cpu().data.numpy()
            offset = offset[0].cpu().data.numpy()#data not data()

            # cond = cond[0][0]
            # offset = offset[0]
            # print(cond)
            # print(np.where(cond > 0.5))
            # idx = torch.gt(cond, 0.5)

            # index_ = np.where(cond > 0.6)
            _idx = np.where(cond > 0.6)
            # idx = np.stack([index_[0], index_[1]], axis=1)
            # offset = offset.numpy()

            idx = np.stack([_idx[0], _idx[1]], axis=1)
            # print('n---------------',idx)
            # exit()
            for id in idx:
                cond_ = cond[id[0], id[1]]
                # print('cond_',cond_, cond_.shape)


                # print('offset', offset, offset.shape)
                # print('offset_', offset[:,id[0],id[1]],offset[:,id[0],id[1]].shape)
                # print('id',id)
                # exit()


                # _x1 = (id[1] * stride)/scale
                # _y1 = (id[0] * stride)/scale
                # _x2 = (id[1] * stride + side_len) / scale
                # _y2 = (id[0] * stride + side_len) / scale

                # ow = _x2 - _x1
                # oh = _y2 - _y1

                offset_ = offset[:, id[0], id[1]]

                # x1 = _x1 + ow*offset_[0]
                # y1 = _y1 + oh*offset_[1]
                # x2 = _x2 + ow*offset_[2]
                # y2 = _y2 + oh*offset_[3]

                x1 = (id[1]*stride + side_len*offset_[0])/scale
                y1 = (id[0]*stride + side_len*offset_[1])/scale
                x2 = (id[1]*stride+side_len + side_len*offset_[2])/scale
                y2 = (id[0]*stride+side_len + side_len*offset_[3])/scale

                pnetbox_.append(np.array([x1, y1, x2, y2, cond_]))
            # print('pnetbox_', pnetbox_, np.array(pnetbox_).shape)



            scale *= 0.7
            _w = int(w*scale)
            _h = int(h*scale)
            img = img.resize((_w, _h))
            len = min(int(_w), int(_h))

        return NMS(np.array(pnetbox_), 0.3, False)
    def _rnet(self, image, pnet_box, side_len):
        img = image
        _pnet_box = Re2Sq(pnet_box)

        _imgset = []
        rnetboxes = []

        # print(_pnet_box, _pnet_box.shape)
        # exit()

        # img = Image.open(image)
        for box in _pnet_box:
            print(box)


            _img = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

            r_img = _img.resize((side_len, side_len))

            img_data = np.array(r_img) / 255-0.5
            # print(img_data,img_data.shape)


            # imgdata = self.img_tensor(_img)
            _imgset.append(img_data)
        # print(np.array(_imgset),np.array(_imgset).shape)
        # print(_imgset)

        _img_data = torch.Tensor(np.array(_imgset))
        # print(_img_data)

        pu_img_data = _img_data.permute(0, 3, 1, 2)


        # img_set = torch.stack(_imgset)
        if self.iscuda == True:
            pu_img_data = pu_img_data.cuda()

        cond, offset = self.rnet(pu_img_data)
        cond = cond.cpu().data.numpy()
        offset = offset.cpu().data.numpy()


        print(cond)

        print(np.stack(np.where(cond > 0.7),axis=1))
        exit()


if __name__ == '__main__':
    pnet_para_path = './parameter/pnet.pkl'
    rnet_para_path = './parameter/rnet.pkl'
    onet_para_path = './parameter/onet.pkl'
    image_path = '/home/ray/datasets/Mtcnn/test/pic/timg2.jpg'

    detector = Detect(pnet_para_path=pnet_para_path, rnet_para_path=rnet_para_path, onet_para_path=onet_para_path, iscuda=True)
    img = Image.open(image_path)
    boxes = detector.detect(img)
    # print('bbb',boxes)
    iDraw = ImageDraw.Draw(img)
    for box in boxes:

        print(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        iDraw.rectangle((int(box[0]), int(box[1]), int(box[2]), int(box[3])), outline='red')
    img.show()





# import torch
# import numpy as np
# import face_net
# from torchvision import transforms
# import utils
# from PIL import Image
# from PIL import ImageDraw
# class Detector:
#
#     def __init__(self, isCuda = True, pnet_param='./paragram/pnet.pt',
#                  rnet_param='./paragram/rnet.pt', onet_param='./paragram/onet.pt'):
#         self.isCuda = isCuda
#         self.pnet = face_net.Pnet()
#         self.rnet = face_net.Rnet()
#         self.onet = face_net.Onet()
#
#         if self.isCuda:
#
#
#             self.pnet.cuda()
#             self.rnet.cuda()
#             self.onet.cuda()
#
#
#         self.pnet.load_state_dict(torch.load(pnet_param))
#         self.rnet.load_state_dict(torch.load(rnet_param))
#         self.onet.load_state_dict(torch.load(onet_param))
#
#         self.pnet.eval()
#         self.rnet.eval()
#         self.onet.eval()
#
#         self._img_transform = transforms.Compose([transforms.ToTensor()])
#     def detect(self,image):
#         pnet_boxes = self._pnet_detect(image)
#         if pnet_boxes.shape[0] == 0:
#             return np.array([])
#         print('P网络框----ok')
#         rnet_boxes = self._rnet_detect(pnet_boxes, image)
#         if rnet_boxes.shape[0] == 0:
#             return np.array([])
#         print('r网络框----ok')
#         onet_boxes = self._onet_detect(rnet_boxes, image)
#         if onet_boxes.shape[0] == 0:
#             return np.array([])
#         print('o网络框----ok')
#
#
#         return onet_boxes
#
#         # return onet_boxes
#
#
#     def _pnet_detect(self,image):
#         boxes=[]
#         img=image
#         w,h=img.size
#         min_side_len = min(w, h)
#
#         scale = 1
#         count=0
#
#         while min_side_len > 12:
#             img_data = self._img_transform(img)
#             if self.isCuda:
#                 img_data = img_data.cuda()
#             img_data.unsqueeze_(0)
#             _cls, _offset = self.pnet(img_data)
#
#             '''print(_cls, _cls.shape)
#             device = 'cuda:0', grad_fn = < SigmoidBackward >) torch.Size([1, 1, 595, 595])
#             print(_offest, _offest.shape)
#             device='cuda:0', grad_fn=<CudnnConvolutionBackward>) torch.Size([1, 4, 595, 595])
#             '''
#
#             cls, offset = _cls[0][0].cpu().data, _offset[0].cpu().data
#
#             '''print(cls, cls.shape)
#             tensor([[0.0009, 0.0009, 0.0009, ..., 0.0010, 0.0010, 0.0010],
#                     [0.0009, 0.0009, 0.0009, ..., 0.0011, 0.0011, 0.0010],
#                     [0.0009, 0.0009, 0.0009, ..., 0.0010, 0.0011, 0.0011],
#                     ...,
#                     [0.0014, 0.0015, 0.0014, ..., 0.0137, 0.0338, 0.0261],
#                     [0.0014, 0.0011, 0.0009, ..., 0.0077, 0.0438, 0.0269],
#                     [0.0011, 0.0010, 0.0012, ..., 0.0193, 0.1236, 0.0439]])
#             torch.Size([595, 595])'''
#
#
#             '''print(offset, offset.shape)
#             tensor([[[2.0649e-01, 2.0649e-01, 2.0649e-01, ..., 2.0681e-01,
#                       2.0426e-01, 2.0291e-01],
#                      [2.0649e-01, 2.0649e-01, 2.0649e-01, ..., 2.0670e-01,
#                       2.0334e-01, 2.0164e-01],
#                      [2.0649e-01, 2.0649e-01, 2.0649e-01, ..., 2.0734e-01,
#                       2.0640e-01, 2.0292e-01],
#                      ...,
#                      [1.2801e-01, 1.2558e-01, 1.4914e-01, ..., 2.6413e-01,
#                       2.4559e-01, 9.8434e-02],
#                      [1.0460e-01, 1.3561e-01, 1.8922e-01, ..., 2.4229e-01,
#                       2.0953e-01, 8.9834e-02],
#                      [1.2472e-01, 1.8638e-01, 2.3591e-01, ..., 3.0332e-01,
#                       1.8824e-01, 1.1457e-01]],
#
#                     [[-6.8426e-02, -6.8426e-02, -6.8426e-02, ..., -6.6802e-02,
#                       -6.7236e-02, -6.6343e-02],
#                      [-6.8426e-02, -6.8426e-02, -6.8426e-02, ..., -6.7088e-02,
#                       -6.4909e-02, -6.5002e-02],
#                      [-6.8426e-02, -6.8426e-02, -6.8426e-02, ..., -7.0906e-02,
#                       -6.8370e-02, -6.8077e-02],
#                      ...,
#                      [-4.3536e-02, -6.3429e-02, -8.3197e-02, ..., -3.2533e-02,
#                       -5.2551e-02, -8.9361e-02],
#                      [-6.6939e-02, -7.1063e-02, -5.0199e-02, ..., 2.2653e-02,
#                       6.3073e-03, -3.7894e-02],
#                      [-6.8528e-02, -3.4867e-02, -1.5526e-02, ..., 3.9639e-02,
#                       -1.1611e-02, -3.5903e-02]],
#
#                     [[-7.9048e-02, -7.9048e-02, -7.9048e-02, ..., -8.0758e-02,
#                       -8.2230e-02, -8.3827e-02],
#                      [-7.9048e-02, -7.9048e-02, -7.9048e-02, ..., -7.8848e-02,
#                       -8.4009e-02, -8.5527e-02],
#                      [-7.9048e-02, -7.9048e-02, -7.9048e-02, ..., -7.8010e-02,
#                       -8.1050e-02, -8.3714e-02],
#                      ...,
#                      [-1.7923e-01, -1.8276e-01, -1.6082e-01, ..., -1.0190e-01,
#                       -7.9041e-02, -1.9877e-01],
#                      [-2.0111e-01, -1.7958e-01, -1.4696e-01, ..., -1.1114e-01,
#                       -1.2443e-01, -2.3903e-01],
#                      [-1.8949e-01, -1.4134e-01, -1.0147e-01, ..., -5.3937e-02,
#                       -1.0101e-01, -2.3596e-01]],
#
#                     [[2.1414e-02, 2.1414e-02, 2.1414e-02, ..., 2.0987e-02,
#                       2.1535e-02, 2.2235e-02],
#                      [2.1414e-02, 2.1414e-02, 2.1414e-02, ..., 2.3029e-02,
#                       2.3272e-02, 2.3234e-02],
#                      [2.1414e-02, 2.1414e-02, 2.1414e-02, ..., 1.9492e-02,
#                       1.9977e-02, 2.0809e-02],
#                      ...,
#                      [2.6329e-02, 5.9889e-03, -1.5483e-02, ..., -1.7519e-02,
#                       8.3837e-04, -4.6049e-03],
#                      [4.5603e-03, -8.6452e-03, -9.7773e-03, ..., 4.7824e-02,
#                       5.8902e-02, 1.8087e-02],
#                      [-6.4614e-03, 1.3221e-02, 2.1965e-02, ..., 6.0331e-02,
#                       9.8738e-02, 6.9605e-03]]])
#             torch.Size([4, 595, 595])'''
#             index_=np.where(cls>0.6)
#             index = np.stack([index_[0], index_[1]], axis=1)
#             offset = offset.numpy()
#
#             # print(index[0])
#             # print(index[1])
#             # print('np.stack',np.stack([index[0], index[1]], axis=1))
#             # print(cls)
#
#             # index = torch.nonzero(torch.gt(cls, 0.6))
#
#
#             for idx in index:
#                 '''print('cls01', cls[idx[0], idx[1]])
#                 cls01
#                 tensor(0.7230)
#                 '''
#
#                 boxes.append([self._box(idx, offset, cls[idx[0], idx[1]], scale)])
#
#
#             scale *= 0.7
#
#             _w = int(w*scale)
#             _h = int(h*scale)
#             img = img.resize((_w, _h))
#             min_side_len = min(_w, _h)
#
#             count+=1
#             boxes_ = np.squeeze(boxes, axis=1)
#
#
#         return utils.NMS(np.array(boxes_), 0.5, mode='UNION')
#     def _box(self, _index, offset, cls,scale, side_len=12,stride=2 ):
#
#         _x1 = (_index[1]*stride)/scale
#
#         _y1 = (_index[0]*stride)/scale
#         _x2 = (_index[1]*stride+side_len)/scale
#         _y2 = (_index[0]*stride+side_len)/scale
#
#         _x1 = float(_x1)
#
#         _y1 = float(_y1)
#         _x2 = float(_x2)
#         _y2 = float(_y2)
#
#         ow = float(_x2-_x1)
#         oh = float(_y2-_y1)
#
#         _offset = offset[:, _index[0], _index[1]]
#
#         x1 = _x1+ow*_offset[0]
#         y1 = _y1+oh*_offset[1]
#         x2 = _x2+ow*_offset[2]
#         y2 = _y2+oh*_offset[3]
#
#
#
#
#         return [x1, y1,x2, y2, cls]
#
#     def _rnet_detect(self, pnet_boxes, image):
#         _img_dataset = []
#         _pnet_boxes=utils.rec2sqr(pnet_boxes)
#         for _box in _pnet_boxes:
#             _x1=int(_box[0])
#             _y1=int(_box[1])
#             _x2=int(_box[2])
#             _y2=int(_box[3])
#             img=image.crop((_x1,_y1,_x2,_y2))
#             img=img.resize((24,24))
#             img_data=self._img_transform(img)
#             _img_dataset.append(img_data)
#
#         img_dataset=torch.stack(_img_dataset)
#         if self.isCuda:
#             img_dataset=img_dataset.cuda()
#
#         _cls,_offset=self.rnet(img_dataset)
#         cls=_cls.cpu().data.numpy()
#         offset=_offset.cpu().data.numpy()
#         #print(cls)
#
#         boxes=[]
#         idxs,_=np.where(cls>0.7)
#
#         for idx in idxs:
#             _box=_pnet_boxes[idx]
#             _x1=int(_box[0])
#             _y1=int(_box[1])
#             _x2=int(_box[2])
#             _y2=int(_box[3])
#
#             ow=_x2-_x1
#             oh=_y2-_y1
#
#             x1=_x1+ow*offset[idx][0]
#             y1=_y1+oh*offset[idx][1]
#             x2=_x2+ow*offset[idx][2]
#             y2=_y2+oh*offset[idx][3]
#
#             boxes.append([x1,y1,x2,y2,cls[idx][0]])
#         return utils.NMS(np.array(boxes),0.7, mode='UNION')
#     def _onet_detect(self,rnet_boxes,image):
#         _img_dataset = []
#         _rnet_boxes = utils.rec2sqr(rnet_boxes)
#         # print(_rnet_boxes)
#         for _box in _rnet_boxes:
#             _x1 = int(_box[0])
#             _y1 = int(_box[1])
#             _x2 = int(_box[2])
#             _y2 = int(_box[3])
#             img = image.crop((_x1, _y1, _x2, _y2))
#             img = img.resize((48, 48))
#             img_data = self._img_transform(img)
#             _img_dataset.append(img_data)
#
#         img_dataset = torch.stack(_img_dataset)
#         if self.isCuda:
#             img_dataset = img_dataset.cuda()
#
#         _cls, _offset = self.onet(img_dataset)
#         cls = _cls.cpu().data.numpy()
#         # print(.data.numpy)
#         offset = _offset.cpu().data.numpy()
#         # print(offset)
#
#         boxes = []
#         idxs, _ = np.where(cls > 0.97)
#         for idx in idxs:
#             _box = _rnet_boxes[idx]
#             _x1 = int(_box[0])
#             _y1 = int(_box[1])
#             _x2 = int(_box[2])
#             _y2 = int(_box[3])
#
#             ow = _x2 - _x1
#             oh = _y2 - _y1
#
#             x1 = _x1 + ow * offset[idx][0]
#             y1 = _y1 + oh * offset[idx][1]
#             x2 = _x2 + ow * offset[idx][2]
#             y2 = _y2 + oh * offset[idx][3]
#             # print(offset[idx])
#
#             boxes.append([x1, y1, x2, y2, cls[idx][0]])
#         return utils.NMS(np.array(boxes), 0.7, mode='isMin')
# if __name__ == '__main__':
#     image_file=r'C:\work\test-080611\cebela\test\timg.jpg'
#     detector=Detector()
#     with Image.open(image_file) as img:
#         boxes=detector.detect(img)
#         imDraw=ImageDraw.Draw(img)
#
#         for box in boxes:
#             x1=int(box[0])
#             y1 = int(box[1])
#             x2 = int(box[2])
#             y2 = int(box[3])
#
#             imDraw.rectangle((x1, y1, x2, y2), outline='red')
#         img.show()









