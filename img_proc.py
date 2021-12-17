import cv2
import numpy as np
import np_utils

def color_quantization(img, k_level):    # src: https://blog.csdn.net/Ibelievesunshine/article/details/105681167
    data = img.reshape((-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(data, k_level, None, criteria, 10, flags)
#     print('compactness：', compactness)
#     print('labels.shape：', labels.shape)
#     print('centers.shape：', centers.shape)
#     print('labels：\n', labels)
#     print('centers：\n', centers)
    
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
#     print('res：\n', res)
    dst = res.reshape((img.shape))
    
    return dst

# create masks for color layers
def seperate_colors(src, blur_size, quantize_level):
    blur = cv2.GaussianBlur(src,(blur_size, blur_size),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    src_quantized = color_quantization(hsv, k_level = quantize_level)
    colors = np_utils.get_colors(src_quantized)
    masks = []
    for i in colors:
        current = cv2.inRange(src_quantized, lowerb = i, upperb = i)
        masks.append(current)
    return colors, masks, src_quantized

## not using them
# def hsv2rgb(hsv):
#     h = hsv[0] / 255
#     s = hsv[1] / 255
#     v = hsv[2] / 255
#     return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
# def median_filter(img,k=3,padding=None):
#     imarray=img
#     height = imarray.shape[0]
#     width = imarray.shape[1]
#     if not padding:
#         edge = int((k - 1) / 2)
#         if height - 1 - edge <= edge or width - 1 - edge <= edge:
#             print("The parameter k is to large.")
#             return None
#         new_arr = np.zeros((height, width), dtype="uint8")
#         for i in range(edge,height-edge):
#             for j in range(edge,width-edge):
#                 new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])# 调用np.median求取中值
#     return new_arr