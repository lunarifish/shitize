import cv2
import img_proc
import numpy as np
import sys
import time
from operator import itemgetter

# load picture
src = cv2.imread(sys.argv[1])
filename = 'assets/output.jpg'
width = src.shape[1]
height = src.shape[0]
print('Source width: {}\nSource height: {}\n'.format(width, height))

# color quantization & masking
quantize_level = int(input('Colors:'))
blur_size = int(input('Blurring intensity:'))
if blur_size % 2 == 0: blur_size += 1
print('\nColor quantization start\nQuantize series: {}\nFilter core size: {}'.format(quantize_level, blur_size))
start_time = time.time()
colors, masks, src_quantized = img_proc.seperate_colors(src, blur_size, quantize_level)
src_quantized = cv2.cvtColor(src_quantized, cv2.COLOR_HSV2RGB)    # for preview, can be disabled
print('Finished ({}s)'.format(time.time() - start_time))

# create a empty canvas to draw result
result = np.zeros((height,width,3), np.uint8)
result.fill(255)

print('Analyzing ellipse size')
shapes_count = 0
ellipses = []
sizes = []
start_time = time.time()
for mask in range(len(masks) - 1):    # repeat the process on every color layers
    # canny
    edge_threshold = 98
    edge_threshold2 = 2.5 * edge_threshold
    edge = cv2.Canny(masks[mask], edge_threshold, edge_threshold2)

    # find closed contours
    edgeFC, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, edgeFC, -1, (255, 255, 255), 4)    # draw thicker lines to fix discontinuous outline
    edge = cv2.GaussianBlur(edge,(blur_size, blur_size),0)    # smooth
    ret, edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    edgeFC, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # drawing ellipses
    for i in range(len(edgeFC)):
        if len(edgeFC[i]) > 4:
            ellipse = cv2.fitEllipse(edgeFC[i])
            ell_size = ellipse[1][0] * ellipse[1][1] # multiply 2 axes's length of the ellipse to get relative size of the ellipse
            ellipses.append([ellipse, colors[mask], ell_size])
            shapes_count += 1
        else: pass
ellipses = sorted(ellipses, key = itemgetter(2), reverse = True) # draw big ellipses at first to avoid smaller ones being covered
print('Finished ({}s)'.format(time.time() - start_time))
print('Total shapes: {}'.format(shapes_count))
print('Drawing shapes to the canvas')
start_time = time.time()
for ellipse in ellipses:
    cv2.ellipse(result, ellipse[0], list(map(int,ellipse[1])), -1)
print('Finished ({}s)'.format(time.time() - start_time))
result = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
cv2.imwrite(filename, result)
print('Result saved as {}'.format(filename))
cv2.imshow('src', src)
cv2.imshow('src_quantized', src_quantized)
cv2.imshow('mask', edge)
cv2.imshow('output', result)

cv2.waitKey(0)
cv2.destroyAllWindows()

## remove convenity defects on contours
## result is WAY TOO terrible someone help me SOS
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
