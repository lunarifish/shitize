import cv2
from img_proc import color_quantization
from img_proc import median_filter
src = cv2.imread('src.jpg')
print(src.shape)

blur_level = 45
blur = cv2.GaussianBlur(src,(blur_level,blur_level),0)

src_quantized= color_quantization(blur, k_level = 8)
print('Color quantization succeed')

edge_threshold = 98
edge_threshold2 = 2.5 * edge_threshold
edge = cv2.Canny(src, edge_threshold, edge_threshold2)
print('1st edge detection succeed')
print('Creating a binary canvas')
edgeFC, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


print('2nd edge detection succeed')
cv2.drawContours(edge, edgeFC, -1, (255, 255, 255), 4)
print('Connecting edges')
edgeFC, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print('3rd edge detection succeed')
for i in range(len(edgeFC)):
    if cv2.contourArea(edgeFC[i]) > 100:
        hull = cv2.convexHull(edgeFC[i],returnPoints=False)
        defects = cv2.convexityDefects(edgeFC[i],hull)
        if defects is not None:
            for j in range(defects.shape[0]):
                s,e,f,d=defects[j,0]
    #             start = tuple(edgeFC[0][s][0])
    #             end = tuple(edgeFC[0][e][0])
                far = tuple(edgeFC[i][f][0])
    #             cv2.line(edge,start,end,(255,255,255))
                cv2.circle(src,far,5,(0,255,0))
print('code=0')
# print(edgeFC)
for i in range(len(edgeFC)):
    if len(edgeFC[i]) > 4:
        ellipse = cv2.fitEllipse(edgeFC[i])
        cv2.ellipse(src, ellipse, (0, 0, 255), 2)
    else: pass

cv2.imshow('src', src)
cv2.imshow('src_quantized', src_quantized)
cv2.imshow('edge', edge)

cv2.waitKey(0)
cv2.destroyAllWindows()