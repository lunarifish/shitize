import cv2
import img_proc
import numpy as np
from np_utils import get_colors
import matplotlib.pyplot as plt

src = cv2.imread('src2.jpg')
width = src.shape[1]
height = src.shape[0]
print(src.shape)

quantize_level = 64
blur_size = 75
colors, masks, src_quantized = img_proc.seperate_colors(src, blur_size, quantize_level)

src_quantized = cv2.cvtColor(src_quantized, cv2.COLOR_HSV2RGB)

blank_image = np.zeros((height,width,3), np.uint8)
blank_image.fill(255)
src_size = width * height
ellipse_size = [1, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]
for size in range(len(ellipse_size)):
    for mask in range(len(masks) - 1):
        edge_threshold = 98
        edge_threshold2 = 2.5 * edge_threshold
        edge = cv2.Canny(masks[mask], edge_threshold, edge_threshold2)
        print('1st edge detection succeed')
        print('Creating a binary canvas')
        edgeFC, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edge, edgeFC, -1, (255, 255, 255), 4)
        edge = cv2.GaussianBlur(edge,(blur_size, blur_size),0)
        ret, edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        edgeFC, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     print(colors)
    #     print(img_proc.hsv2rgb(list(map(int,colors[mask]))))
        print(list(map(int,colors[mask])))
        for i in range(len(edgeFC)):
            if len(edgeFC[i]) > 4:
                ellipse = cv2.fitEllipse(edgeFC[i])
                ell_size = ellipse[1][0] * ellipse[1][1]
                if ell_size < ellipse_size[size] * src_size and ell_size > ellipse_size[size + 1] * src_size:
                    cv2.ellipse(blank_image, ellipse, list(map(int,colors[mask])), -1)
            else: pass

blank_image = cv2.cvtColor(blank_image, cv2.COLOR_HSV2RGB)
cv2.imshow('src', src)
cv2.imshow('src_quantized', src_quantized)
# cv2.imshow('mask', edge)
cv2.imshow('output', blank_image)


cv2.waitKey(0)
cv2.destroyAllWindows()


# for i in range(len(edgeFC)):
#     if cv2.contourArea(edgeFC[i]) > 100:
#         points_far = []
#         hull = cv2.convexHull(edgeFC[i],returnPoints=False)
#         defects = cv2.convexityDefects(edgeFC[i],hull)
#         if defects is not None:
#             for j in range(defects.shape[0]):
#                 s,e,f,d=defects[j,0]
#                 start = tuple(edgeFC[i][s][0])
#                 end = tuple(edgeFC[i][e][0])
#                 far = tuple(edgeFC[i][f][0])
#                 cv2.line(src,start,end,(255,255,255))
#                 if d > 300:
#                     points_far.append(far)
#                     cv2.circle(src,far,5,(0,255,0))
#                 for k in range(len(points_far) - 1):
#                     cv2.line(edge,points_far[k],points_far[k + 1],(255,255,255))
