import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import IMG_FORMATS, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes

from cv_bridge import CvBridge

class yolov5_demo():
    def __init__(self,  weights,
                        data,
                        imagez_height,
                        imagez_width,
                        conf_thres,
                        iou_thres,
                        max_det,
                        device,
                        view_img,
                        classes,
                        agnostic_nms,
                        line_thickness,
                        half,
                        dnn
                        ):
        self.weights = weights
        self.data = data
        self.imagez_height = imagez_height
        self.imagez_width = imagez_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn

        self.s = str()
        
        self.define_cvBridge()
        self.load_model()

    def define_cvBridge(self):
        self.bridge = CvBridge()
    
    def load_model(self):
        imgsz = (self.imagez_height, self.imagez_width)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        source = 0
        # Dataloader
        webcam = True
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            # bs = len(dataset)  # batch_size
        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=self.half)  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    # callback ==========================================================================

    def image_callback(self, image:Image):
        im = self.bridge.imgmsg_to_cv2(image, "bgr8")

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        save_dir = "runs/detect/exp7"
        path = ['0']

        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            # seen += 1
            webcam = True
            if webcam:  # batch_size >= 1
                p = 0
                frame = 0
                # im0 = im0s[i].copy()
                im0 = im
                self.s += f'{i}: '
            # else:
            #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            self.s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    save_conf = False
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    if self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{self.names[c]} {self.conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if save_crop:
                        #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


            # Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        LOGGER.info(f'{self.s}Done. ({t3 - t2:.3f}s)')


class yolov5_ros(Node):
    def __init__(self):
        super().__init__('yolov5_ros')
        self.pub_bbox = self.create_publisher(BoundingBoxes, 'bounding_boxes', 10)
        self.sub_image = self.create_subscription(Image, 'image', self.image_callback,10)

        # parameter
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

        self.declare_parameter('weights', ROOT + 'config/yolov5s.cfg')
        self.declare_parameter('data', ROOT + 'config/coco128.yaml')
        self.declare_parameter('imagez_height', 640)
        self.declare_parameter('imagez_width', 640)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 1000)
        self.declare_parameter('device', '0')
        self.declare_parameter('view_img', False)
        self.declare_parameter('classes', None)
        self.declare_parameter('agnostic_nms', False)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('half', False)
        self.declare_parameter('dnn', False)

        self.weights = self.get_parameter('weights').value
        self.data = self.get_parameter('data').value
        self.imagez_height = self.get_parameter('imagez_height').value
        self.imagez_width = self.get_parameter('imagez_width').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.view_img = self.get_parameter('view_img').value
        self.classes = self.get_parameter('classes').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.half = self.get_parameter('half').value
        self.dnn = self.get_parameter('dnn').value

        # imgsz = (self.imagez_height,self.imagez_width)
        self.yolov5 = yolov5_demo(self.weights,
                                self.data,
                                self.imagez_height,
                                self.imagez_width,
                                self.conf_thres,
                                self.iou_thres,
                                self.max_det,
                                self.device,
                                self.view_img,
                                self.classes,
                                self.agnostic_nms,
                                self.line_thickness,
                                self.half,
                                self.dnn)

        


    def image_callback(self, image:Image):
        self.yolov5.image_callback(self, image)


def ros_main(args=None):
    rclpy.init(args=args)
    yolov5_node = yolov5_ros()
    rclpy.spin(yolov5_node)
    yolov5_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    ros_main()