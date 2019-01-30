import numpy as np
b = np.random.randint(1,100,100)
flag = 1000
print(b)
for i in b:
    if flag > i:
        flag = i
print('flag', flag)
