import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def find_coef(show):
    image_name = "data/fisheye2.jpg"
    out_folder = "fisheye_test"
    if not os.path.exists(out_folder):
        # shutil.rmtree(output_folder)  # delete output folder
        os.makedirs(out_folder)  # make new output folder

    DIM=(1406, 977)
    K=np.array([[158.90734266803284, 0.0, 237.55632924615173], 
                [0.0, 159.1042171794561, 133.46049082210746], 
                [0.0, 0.0, 1.0]])
    D=np.array([[-0.15062360914219375], [-0.005569763871304728], [0.05244409490545496], [-0.027976280567001712]])

    K = np.array([[  689.21,     0.  ,  1295.56],
                [    0.  ,   690.48,   600.17],
                [    0.  ,     0.  ,     1.  ]])

    img = cv2.imread(image_name)
    plt.figure(1)

    # plt.imshow(img)
    # plt.show()

    step = 10
    j = 0
    koef = 0.001
    while (j == 0):
        i = 0
        while(i == 0):
            K_copy = K.copy() 
            K_copy[0, 0] = K_copy[0, 0] + step * 70
            K_copy[1, 1] = K_copy[1, 1] + step * (-10)
            K_copy[0, 2] = K_copy[0, 2] + step * (-89)
            K_copy[1, 2] = K_copy[1, 2] + step * (-5)            


            DIM_copy = (int(DIM[0] * 1.2), int(DIM[1] * 1.2))
            
            Knew = K_copy.copy() 
            # Knew[0 , 0] = Knew[0 , 0] * 0.5
            # Knew[1 , 1] = Knew[1 , 1] * 0.5

            D_copy = D.copy()
            D_copy[0 , 0] = D_copy[0 , 0] * koef * 190
            # D_copy[1 , 0] = D_copy[1 , 0] * koef

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_copy, D_copy, np.eye(3), Knew, DIM_copy, cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            cv2.imwrite(out_folder + "/undistort_" + str(j) + "_" + str(i) + ".jpg", undistorted_img)
            if show:
                plt.imshow(undistorted_img)
                plt.show()

            print(K_copy)
            print(D_copy)
            print(DIM_copy)
            i = i + 1
        j = j + 1

    cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # Knew = K.copy()
    # Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]

    ################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='', help='path to images')
    parser.add_argument('--output', type=str, default='', help='path to images')
    parser.add_argument('--show', type=bool, default=False, help='Show images')
    opt = parser.parse_args()
    print(opt)

    find_coef(opt.show)