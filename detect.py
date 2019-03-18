import argparse

from detect_images import *
from detect_video import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='', help='path to images')
    parser.add_argument('--video', type=str, default='', help='path to images')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45, help='iou threshold for non-maximum suppression')
    parser.add_argument('--coco-max-entity', type=int, default=80, help='Max entity from data/coco.names which will be drown on image')
    parser.add_argument('--frame-step', type=int, default=0, help='Frame step for videofile in milliseconds. 0 - all frames')
    parser.add_argument('--show-image', type=bool, default=False, help='Each frame will be shown')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if not opt.images == "":
            detect_images(
                opt.cfg,
                opt.weights,
                opt.images,
                img_size=opt.img_size,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                save_images=True
            )
        if not opt.video == "":
            detect_video(
                opt.cfg,
                opt.weights,
                opt.video,
                img_size=opt.img_size,
                conf_thres=opt.conf_thres,
                nms_thres=opt.nms_thres,
                coco_max_entity=opt.coco_max_entity,
                frame_step=opt.frame_step,
                show_image=opt.show_image,
                save_video=True,
            )

        if opt.images == "" and opt.video == "":
            print("Specify, please, images or video!")
