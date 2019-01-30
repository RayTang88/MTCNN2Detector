import os
def Change_cond(size):

    img_path = '/home/ray/datasets/Mtcnn/img_celeba_dataset{}'.format(size)
    positive_txt = open(os.path.join(img_path, 'positive_txt'))
    negative_txt = open(os.path.join(img_path, 'negative_txt'))

    lines = positive_txt.readline()

    for
