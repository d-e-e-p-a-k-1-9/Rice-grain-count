import cv2
import pandas as pd
from scipy import ndimage
import numpy as np
from skimage import feature, measure
from skimage.segmentation import watershed


def cv(img):
    ## objective 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    kernelSize = 3
    opIterations = 2
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    output_adapthresh = cv2.adaptiveThreshold (blur, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 451, -30)
    binaryImage = cv2.morphologyEx( output_adapthresh, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101 )

    # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Watershed_Algorithm_Marker_Based_Segmentation.php idea of using watershed algorithm for separating close contours was used with the help of this blog
    dist_trans = ndimage.distance_transform_edt(binaryImage)
    local_max = feature.peak_local_max(dist_trans, min_distance=30)
    local_max_mask = np.zeros(dist_trans.shape, dtype=bool)
    local_max_mask[tuple(local_max.T)] = True
    cont = watershed(-dist_trans, measure.label(local_max_mask), mask=binaryImage) 

    ## objective 2
    (cnt, a) = cv2.findContours(output_adapthresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = []
    for i in cnt:
        if cv2.contourArea(i) < 1300:
            count.append(i)

    return cont.max(), len(count)

broken_rice = []
file = []
total_rice = []

for i in range(1,6):
    img = cv2.imread(r"C:\Users\laxma\Downloads\data\data\test\image_{}.jpg".format(i))
    tr, br = cv(img)
    total_rice.append(tr)
    broken_rice.append(br)
    file.append('image_{}.jpg'.format(i))

list_tup = list(zip(file, total_rice, broken_rice))
df = pd.DataFrame(list_tup, columns=['file_name', 'total_rice_grain', 'total_broken_rice_grain'])
print(df)
df.to_csv('Deepak_BE19B019_cv.csv', index=False)