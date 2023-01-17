import cv2 as cv
import numpy as np
import glob

def hist(img):
    img_convert = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    l, a, b = cv.split(img_convert)

    hist, _ = np.histogram(l.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = np.uint8(cdf/cdf.max()*255)
    equilized = cdf_normalized[l]
    equilized = cv.merge([equilized,a,b])
    hist_final = cv.cvtColor(equilized, cv.COLOR_Lab2BGR)
    
    return hist_final

def adap_hist(img, n = 8):
    img = img.copy()

    h, w, _ = img.shape
    sh, sw = h//n, w//n
    for i in range(0, h, sh):
        for j in range(0, w, sw):
            img[i:i+sh, j:j+sw] = hist(img[i:i+sh, j:j+sw])
    return img



img_array = []
for filename in glob.glob('adaptive_hist_data/adaptive_hist_data/*.png'):
    img = cv.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc(*'DIVX'), 1, size)



for i in range(len(img_array)):
    out.write(img_array[i])
out.release()



vid = cv.VideoCapture('output.avi')


fourcc = cv.VideoWriter_fourcc(*'XVID')
out_hist = cv.VideoWriter('Histogram.avi', fourcc, 1.0, (1224,370))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out_adap_hist = cv.VideoWriter('Adaptive Histogram.avi', fourcc, 1.0, (1224,370))

while (vid.isOpened()):
    ret, frame = vid.read()


    # print(frame.shape)
    

    if ret == True:
            cv.imshow('original', frame)
    else:
        break

    
    histogram = hist(frame)

    adaptive_histogram = adap_hist(frame)

    cv.imshow('histogram equalized', histogram)
    cv.imshow('adaptive histogram equalization', adaptive_histogram)
    if cv.waitKey(500) & 0xFF==ord("d"):
        break

    out_hist.write(histogram)
    out_adap_hist.write(adaptive_histogram)



vid.release()
out.release()
out_hist.release()
out_adap_hist.release()



