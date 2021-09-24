"""
Make updated body shape from updated segmentation
"""

import os
import numpy as np
import cv2
import time
import sys
import PIL.ImageOps

from numpy import asarray
from PIL import Image

(cv_major, _, _) = cv2.__version__.split(".")
if cv_major != '4' and cv_major != '3':
    print('doesnot support opencv version')
    sys.exit()

def add_margin_image(pil_img, top, right, bottom, left):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new("RGB", (new_width, new_height) ,(255,255,255))
    result.paste(pil_img, (left, top))
    return result

def add_margin_image_mask(pil_img, top, right, bottom, left):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new("RGB", (new_width, new_height) ,(0,0,0))
    result.paste(pil_img, (left, top))
    return result


# @TODO this is too simple and pixel based algorithm
def body_detection(image, seg_mask):
    # binary thresholding by blue ?
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)# To convert BGR (Blue Green Red) color space to HSV (Hue Saturation
    # Value) color space which is mostly used for object detection.
    lower_blue = np.array([0, 0, 120])# upper_blue - lower_blue is actual covering all colors as hue values can vary from 0 to 180.
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)# Creating a mask by filtering colors which have hsv values between [0,0,120] to [180,38,255]
    result = cv2.bitwise_and(image, image, mask=mask)

    # binary threshold by green ?
    b, g, r = cv2.split(result)
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)

    # at least original segmentation is FG
    mask[seg_mask] = 1#doubtful

    return mask


def make_body_mask(data_dir, seg_dir, image_name, mask_name, save_dir=None):
    print(image_name)

    # define paths
    img_pth = os.path.join(data_dir, image_name)
    seg_pth = os.path.join(seg_dir, mask_name)

    mask_path = None
    if save_dir is not None:
        mask_path = os.path.join(save_dir, mask_name)

    # Load images
    img = cv2.imread(img_pth)
    # segm = Image.open(seg_pth)
    # the png file should be 1-ch but it is 3 ch ^^;
    gray = cv2.imread(seg_pth, cv2.IMREAD_GRAYSCALE)
    _, seg_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    body_mask = body_detection(img, seg_mask)
    
    body_mask = body_mask + seg_mask
    body_mask[seg_mask] = 1# Doubtful
    #body_mask = body_mask.crop((0,0,192,256))
    cv2.imwrite(mask_path, body_mask)
    body_mask = Image.open(mask_path)
    print(body_mask.size)
    body_mask = body_mask.crop((0,0,192,256))
    body_mask.save(mask_path)
    print(body_mask.size)
    
def make_cloth_mask(data_dir,cloth_name, cloth_mask_name, save_dir=None):
    print(cloth_name)
    
    cloth_pth = os.path.join(data_dir,cloth_name)
    
    cloth_mask_path = None
    
    if save_dir is not None:
        
        cloth_mask_path = os.path.join(save_dir,cloth_mask_name)
        cloth_mask_path_2 = os.path.join(save_dir,'000003_1.jpg')
        
    #gray = cv2.imread(img_pth,cv2.IMREAD_GRAYSCALE)
    #_,cloth_mask = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
    cloth_mask = Image.open(cloth_pth)
    cloth_mask = PIL.ImageOps.invert(cloth_mask)
    cloth_mask.save(cloth_mask_path)
    IMG = Image.open(cloth_mask_path_2)
    print(IMG.size)
    
    img = cv2.imread(cloth_mask_path,0)
    img = cv2.medianBlur(img,5)
    
    #hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    gray = cv2.imread(cloth_mask_path,cv2.IMREAD_GRAYSCALE)
    gray = cv2.medianBlur(gray,21)
    #_,cloth_mask = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #_,cloth_mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #
    cloth_mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,901,2)
    cv2.imwrite(cloth_mask_path,cloth_mask)
    cloth_mask = Image.open(cloth_mask_path)
    cloth_mask_array = asarray(cloth_mask)
    print(cloth_mask_array[255])
    result = np.where(cloth_mask_array == 255)
    #result = np.where(cloth_mask_array >= cloth_mask_array.mean)
    
    
    left  = min(result[1])
    upper = min(result[0])
    right = max(result[1])              
    lower = max(result[0])
    
    left  = min(result[1]) if ( (min(result[1])-4) > 0 ) else (4)
    upper = min(result[0]) if ( (min(result[0])-4) > 0 ) else (4)
    right = max(result[1]) if ( (max(result[1])+4) < 191 ) else (187)             
    lower = max(result[0]) if ( (max(result[0])+4) < 255 ) else (251)
    print(left,upper,right,lower)
    result_col_index_list = result[1].tolist()
    
    
    
    cloth_mask_cropped = Image.open("data/test/cloth-mask/Sample_1.jpg")
    cloth_mask_cropped = cloth_mask_cropped.crop((left-4,upper-4,right+4,lower+4))# left, upper, right, lower
    cloth_mask_cropped.save("data/test/cloth-mask/Sample_1.jpg")
    cloth_mask_cropped_resized = Image.open("data/test/cloth-mask/Sample_1.jpg")
    cloth_mask_cropped_resized = cloth_mask_cropped.resize((192,256),resample = Image.NEAREST)
    cloth_mask_cropped_resized.save("data/test/cloth-mask/Sample_1.jpg")
    
    cloth_cropped = Image.open("data/test/cloth/Sample_1.jpg")
    cloth_cropped = cloth_cropped.crop((left-4,upper-4,right+4,lower+4))# left, upper, right, lower
    cloth_cropped.save("data/test/cloth/Sample_1.jpg")
    cloth_cropped_resized = Image.open("data/test/cloth/Sample_1.jpg")
    cloth_cropped_resized = cloth_cropped.resize((192,256),resample = Image.NEAREST)
    cloth_cropped_resized.save("data/test/cloth/Sample_1.jpg")
    
    
    image_mask_path = "data/test/image-mask/Sample_0.png"
    
    image_mask = Image.open(image_mask_path)
    image_mask_array = asarray(image_mask)
    print(image_mask_array[255])
    result = np.where(image_mask_array == 255)
    #result = np.where(image_mask_array >= image_mask_array.mean)
    
    
    #left  = min(result[1]) if ( (min(result[1])-24) > 0 ) else (24)
    #upper = min(result[0]) if ( (min(result[0])-4)  > 0 ) else (4)
    #right = max(result[1]) if ( (max(result[1])+24) < 191 ) else (167)             
    #lower = max(result[0]) if ( (max(result[0])+4)  < 255 ) else (251)
    
    
    print(left,upper,right,lower)
    result_col_index_list = result[1].tolist()
    
    top = 5
    right = 5
    bottom = 5
    left = 5
    color = (255,255,255)
    
    image_mask_padded = Image.open("data/test/image-mask/Sample_0.png")
    image_mask_padded = add_margin_image_mask(image_mask_padded, top, right, bottom, left)
    image_mask_padded.save("data/test/image-mask/Sample_0.png")
    image_mask_padded_resized = Image.open("data/test/image-mask/Sample_0.png")
    image_mask_padded_resized = image_mask_padded.resize((192,256),resample = Image.NEAREST)
    image_mask_padded_resized.save("data/test/image-mask/Sample_0.png")
    
    
    image_padded = Image.open("data/test/image/Sample_0.jpg")
    image_padded = add_margin_image(image_padded, top, right, bottom, left)
    image_padded.save("data/test/image/Sample_0.jpg")
    image_padded_resized = Image.open("data/test/image/Sample_0.jpg")
    image_padded_resized = image_padded.resize((192,256),resample = Image.BILINEAR)
    image_padded_resized.save("data/test/image/Sample_0.jpg")
    
    image_parse_new_padded = Image.open("data/test/image-parse-new/Sample_0.png")
    image_parse_new_padded = add_margin_image(image_parse_new_padded, top, right, bottom, left)
    image_parse_new_padded.save("data/test/image-parse-new/Sample_0.png")
    image_parse_new_padded_resized = Image.open("data/test/image-parse-new/Sample_0.png")
    image_parse_new_padded_resized = image_parse_new_padded.resize((192,256),resample = Image.BILINEAR)
    image_parse_new_padded_resized.save("data/test/image-parse-new/Sample_0.png")
    
    
#    image_mask_cropped = Image.open("data/test/image-mask/Sample_0.png")
#    image_mask_cropped = image_mask_cropped.crop((left-24,upper-4,right+24,lower+4))# left, upper, right, lower
#    image_mask_cropped.save("data/test/image-mask/Sample_0.png")
#    image_mask_cropped_resized = Image.open("data/test/image-mask/Sample_0.png")
#    image_mask_cropped_resized = image_mask_cropped.resize((192,256),resample = Image.NEAREST)
#    image_mask_cropped_resized.save("data/test/image-mask/Sample_0.png")
#    
#    
#    image_cropped = Image.open("data/test/image/Sample_0.jpg")
#    image_cropped = image_cropped.crop((left-24,upper-4,right+24,lower+4))# left, upper, right, lower
#    image_cropped.save("data/test/image/Sample_0.jpg")
#    image_cropped_resized = Image.open("data/test/image/Sample_0.jpg")
#    image_cropped_resized = image_cropped.resize((192,256),resample = Image.BILINEAR)
#    image_cropped_resized.save("data/test/image/Sample_0.jpg")
#    
#    image_parse_new_cropped = Image.open("data/test/image-parse-new/Sample_0.png")
#    image_parse_new_cropped = image_parse_new_cropped.crop((left-24,upper-4,right+24,lower+4))# left, upper, right, lower
#    image_parse_new_cropped.save("data/test/image-parse-new/Sample_0.png")
#    image_parse_new_cropped_resized = Image.open("data/test/image-parse-new/Sample_0.png")
#    image_parse_new_cropped_resized = image_parse_new_cropped.resize((192,256),resample = Image.BILINEAR)
#    image_parse_new_cropped_resized.save("data/test/image-parse-new/Sample_0.png")

    
    
    #cloth_mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,121,2)
    #cv2.imwrite(cloth_mask_path,cloth_mask)
    
    # Otsu's thresholding
    #_,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(img,(50,50),0)
    #_,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imwrite(cloth_mask_path,th3)
    


def main():
    
    seconds_start = time.time()
    
    root_dir = "data/"
    # define pdata/viton_data/"
    mask_folder = "image-mask"
    cloth_mask_folder = "cloth-mask"
    seg_folder = "image-parse-new"
    

    # data_mode = "train"
    data_mode = "test"
    image_folder = "image"
    cloth_folder = "cloth"

    image_dir = os.path.join(os.path.join(root_dir, data_mode), image_folder)
    cloth_dir = os.path.join(os.path.join(root_dir, data_mode), cloth_folder) 
    seg_dir = os.path.join(os.path.join(root_dir, data_mode), seg_folder)

    image_list = sorted(os.listdir(image_dir))
    cloth_list = sorted(os.listdir(cloth_dir))
    seg_list = sorted(os.listdir(seg_dir))
    

    mask_dir = os.path.join(os.path.join(root_dir, data_mode), mask_folder)
    cloth_mask_dir = os.path.join(os.path.join(root_dir, data_mode), cloth_mask_folder)
    
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
        
    if not os.path.exists(cloth_mask_dir):
        os.makedirs(cloth_mask_dir)

    #for each in zip(image_list, seg_list):
     #   make_body_mask(image_dir, seg_dir, each[0], each[1], mask_dir)
    #make_body_mask(image_dir, seg_dir, '000001_0.jpg',
    #              '000001_0.png',mask_dir)
    
    #make_cloth_mask(cloth_dir,"001744_1.jpg","001744_1.jpg",cloth_mask_dir)
    
    make_body_mask(image_dir, seg_dir, 'Sample_0.jpg',
                'Sample_0.png',mask_dir)
    
    make_cloth_mask(cloth_dir,"Sample_1.jpg","Sample_1.jpg",cloth_mask_dir)
        
    
    #for each in zip(cloth_list):
     #   make_cloth_mask(cloth_dir,each,each,cloth_mask_dir)
        
    seconds_end = time.time()    
    
    print(seconds_end-seconds_start)


if __name__ == '__main__':
    main()
