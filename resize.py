import os 
import cv2
import numpy as np
import csv
from tqdm import tqdm

def get_name_from_csv():
    #import csv
    csv_file = './ISIC-2017_Validation_Part3_GroundTruth.csv'
    reader = open(csv_file)
    name_list = []
    for index, i in enumerate(reader):
        if index == 0:
            continue
        #print(i)
        name = i.split(',')[0] + '.jpg'
        name_list.append(name)
        
    return name_list

def resize_dataset(test_img_path,test_gt_path,output_image_path,output_gt_path,name_list,shape):
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_gt_path, exist_ok=True)

    
    for name in tqdm(os.listdir(test_img_path)):
        img = cv2.imread(test_img_path+name)
        #true_image_name = name.split('_')[0] + '_' + name.split('_')[1] + '.jpg' 
        mask_name = name.replace('.jpg','_segmentation.png')
        mask = cv2.imread(test_gt_path+mask_name)

        img = cv2.resize(img,shape)
        mask= cv2.resize(mask,shape)

        cv2.imwrite(output_image_path+name,img)
        cv2.imwrite(output_gt_path+mask_name,mask)

def run_resize():
    #shape_list = [(128,128),(192,192),(224,224),(256,256)]
    #for shape in shape_list:
        shape = (128,128)
        test_img_path = './train_img/'
        test_gt_path = './train_gt/'
    
        output_img_path = './train_{}_img/'.format(shape[0])
        output_gt_path = './train_{}_gt/'.format(shape[0])
    
        name_list = get_name_from_csv()
    
    
        resize_dataset(test_img_path, test_gt_path, output_img_path, output_gt_path, name_list, shape)




if __name__ == '__main__':
    run_resize()