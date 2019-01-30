
import re
import os
class A():
    def __init__(self,size):
        self.size = size
        self.path = '/home/ray/datasets/Mtcnn/img_celeba_dataset/{}'.format(self.size)

    def b(self):
        positive_count = 0
        negative_count = 0

        # positive_txt = open(os.path.join(self.path, 'positive.txt'), 'rb')
        negative_txt = open(os.path.join(self.path, 'negative111.txt'), 'rb')
        # part_txt = open(os.path.join(self.path, 'part.txt'), 'rb')
        # first_line = positive_txt.readline()



        off_p = -50
        off_n = -50
        while True:
            # positive_txt.seek(off_p, 2)
            negative_txt.seek(off_n, 2)  # seek(off, 2)表示文件指针：从文件末尾(2)开始向前50个字符(-50)
            # p_lines = positive_txt.readlines()
            n_lines = negative_txt.readlines()  # 读取文件指针范围内所有行
            # if len(p_lines) >= 2:
            #     # global positive_count# 判断是否最后至少有两行，这样保证了最后一行是完整的
            #     p_last_line = p_lines[-1]
            #     positive_count = re.sub('\D', '', p_last_line.decode()) # 取最后一行
            #     break
            if len(n_lines) >= 2:
                # global negative_count# 判断是否最后至少有两行，这样保证了最后一行是完整的
                n_last_line = n_lines[-1]
                negative_count = int(re.findall(r'(\d+\w+).jpg', n_last_line.decode())[0]) # 取最后一行
                print(negative_count)
                break
            off_p *= 2
            off_n *= 2



        negative_txt.close()
        return negative_count
        # positive_txt.close()
    def f(self):
        negative_txt = open(os.path.join(self.path, 'negative111.txt'), 'w')

        for i in range(30):
            negative_txt.write('positive/{}.jpg 你好\n'.format(i))
        negative_txt.close()
    def g(self):
        negative_count = self.b()+1
        negative_txt = open(os.path.join(self.path, 'negative111.txt'), 'a+')

        for i in range(10):
            negative_txt.write('positive/{}.jpg 你好\n'.format(negative_count))
            negative_count += 1
        negative_txt.close()


if __name__ == '__main__':
    a = A(12)
    a.f()
    a.g()

    # d = a.b(12)

