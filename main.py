import cv2
import numpy as np
import pandas as pd
import os
import time

start_time = time.time()
train_path,train_dirs,train_files = next(os.walk("data/Train"))

out = 'A B F I N S'.split()
temp = ['C', 'D', 'E', 'H', 'L', 'M', 'O', 'P', 'R', 'T', 'U', 'W']

for i in out:
    
    char_path,char_dirs,char_files = next(os.walk(train_path+"/"+i))
    df = pd.DataFrame(columns=['pixel'+str(i) for i in range(28*28*3)]+['label'])
    
    for j in range(len(char_files)//2):
        img = cv2.imread(char_path+"/"+char_files[j])
        img = cv2.resize(img,(28,28))
        img_arr = np.asarray(img)
        img_arr = np.append(img_arr,[i])
        df.loc[j] = img_arr
        
    df.to_csv(r'Modified data/Train3/'+i+'.csv',index=False)
    
end_time = time.time()
print(end_time-start_time)
