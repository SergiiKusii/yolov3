import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np

# You should replace these 3 lines with the output in calibration step
DIM=(478, 276)
K=np.array([[158.90734266803284, 0.0, 237.55632924615173], [0.0, 159.1042171794561, 133.46049082210746], [0.0, 0.0, 1.0]])
D=np.array([[-0.15062360914219375], [-0.005569763871304728], [0.05244409490545496], [-0.027976280567001712]])
def undistort(img_path):
    img = cv2.imread(img_path)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    undistort("data/dashboard1.jpg")