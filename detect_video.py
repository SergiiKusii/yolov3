import cv2
import argparse
import shutil
import time
import matplotlib.pyplot as plt

from pathlib import Path
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

def detect_video(
        cfg,
        weights,
        video_path,
        output_folder='output',  # output folder
        img_size=416,
        conf_thres=0.3,
        nms_thres=0.45,
        coco_max_entity=80,
        save_txt=True,
        save_video=True,
        show_image=False
):
    device = torch_utils.select_device()
    if not os.path.exists(output_folder):
        # shutil.rmtree(output_folder)  # delete output folder
        os.makedirs(output_folder)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        if weights.endswith('yolov3.pt') and not os.path.exists(weights):
            if (platform == 'darwin') or (platform == 'linux'):
                os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Get classes and colors
    classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

    # load video
    capture = cv2.VideoCapture(video_path)
    size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    output_name = str(Path(output_folder) / Path(video_path).name)
    print("Processing video " + video_path + " ...")
    video_output = cv2.VideoWriter(output_name, codec, 10.0, size)

    plt.figure(1)
    handled = False
    frame_num = 0
    t_start = time.time()
    frames_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    step_ms = 500
    while (capture.isOpened()):
        handled = True        
        t = time.time()
        frame_num = frame_num + 1
        frame_sec = capture.get(cv2.CAP_PROP_POS_MSEC)
        frame_precentage = capture.get(cv2.CAP_PROP_POS_AVI_RATIO)
        print('Frame %g/%g (%gms %.5f pr): ' % (frame_num, frames_count, frame_sec, frame_precentage), end='')

        ret, frame = capture.read()

        # if not frame_num % 10 == 0:
        #     print("continue")
        #     continue

        if not ret:
            break

        # Get detections
        im0 = frame
        img = img0_to_img(im0, img_size)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            print ("\nONNX_EXPORT")
            return
        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            unique_classes = detections[:, -1].cpu().unique()
            for c in unique_classes:
                n = (detections[:, -1].cpu() == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                if cls > coco_max_entity:
                    continue

                if save_txt:  # Write to file
                    with open(output_name + '.txt', 'a') as file:
                        file.write('%g %g %g %g %g %g\n' %
                                   (x1, y1, x2, y2, cls, cls_conf * conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])

        video_output.write(frame)

        dt = time.time() - t
        print('Done. (%.3fs)' % dt)

        if show_image:
            plt.imshow(frame)
            plt.show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if handled and save_video:
        capture.release()
        video_output.release()
        cv2.destroyAllWindows()
        print("Saved to " + output_name)

    dt = time.time() - t_start
    print('Done. (%.3fs) with %s' % (dt, video_path))    

