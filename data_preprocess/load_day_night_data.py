import glob
import numpy as np
import os
import pandas as pd
import cv2


names = ['dirty',
'daylight',
'night',
'sunrisesunset',
'dawndusk',
'sunny',
'clouds',
'fog',
'storm',
'snow',
'warm',
'cold',
'busy',
'beautiful',
'flowers',
'spring',
'summer',
'autumn',
'winter',
'glowing',
'colorful',
'dull',
'rugged',
'midday',
'dark',
'bright',
'dry',
'moist',
'windy',
'rain',
'ice',
'cluttered',
'soothing',
'stressful',
'exciting',
'sentimental',
'mysterious',
'boring',
'gloomy',
'lush']

file_path = './data_preprocess/imageAlignedLD/images/'
tsv_file_path = './data_preprocess/imageAlignedLD/annotations/annotations.tsv'
img_list =  glob.glob(os.path.join(file_path, '*', '*.jpg'))
img_list.sort(reverse=True)
file = pd.read_csv(tsv_file_path, sep ='\t', names=names)

get_value = file['night']

save_path = './data_preprocess/rgb/'
save_label_path = './data_preprocess/label/'
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_label_path, exist_ok=True)

for idx, value in enumerate(get_value):
    file_name = file.index[idx]
    
    value = value.split(',')
    confidence = float(value[0])

    path_list = img_list[idx].split('/') # ./test/imageAlignedLD/images/00017660/20120504_022444.jpg 90000014/97.jpg
    image_path = path_list[0] + '/' + path_list[1] + '/' + path_list[2] + '/' + path_list[3] + '/' + '/' + file_name
    

    
    img = cv2.imread(image_path)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)

    cv2.imwrite(save_path + str(idx) + '.png', img)
    
    with open(save_label_path + str(idx) +'.txt', "w") as f:
        if confidence >= 0.3:
            f.write(str(confidence))
        else:
            f.write(str(confidence))

    # print(glob.glob(os.path.join(save_label_path, '*.txt')))

        
    



