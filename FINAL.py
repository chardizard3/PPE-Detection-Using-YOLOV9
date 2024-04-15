import cv2
import PySimpleGUI as sg
import numpy as np
import torch
import time
import RPi.GPIO as GPIO
from gpiozero import Buzzer

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

#breadboard setup

GPIO.setmode(GPIO.BCM)
red = 40
green = 38
blue = 36
motor = 32

GPIO.setup(red, GPIO.OUT)
GPIO.setup(green, GPIO.OUT)
GPIO.setup(blue, GPIO.OUT)
pwm = GPIO.PWM(motor, 50)
buzzer = Buzzer(26)

Freq = 100

RED = GPIO.PWM(red, Freq)
GREEN = GPIO.PWM(green, Freq)
BLUE = GPIO.PWM(blue, Freq)

@smart_inference_mode()
def run(
        weights='/Users/richardpalabasan/Desktop/ppe/yolov9/latest.pt', source='0', 
        data='/Users/richardpalabasan/Desktop/ppe/yolov9/PPE-5/data.yaml', imgsz=(416, 416), 
        conf_thres=0.1, iou_thres=0.45, max_det=1000, device='cpu', classes=None,
        agnostic_nms=False, augment=False, visualize=False, line_thickness=3, hide_labels=False,
        hide_conf=True, half=False, dnn=False, vid_stride=1):
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # GUI layout
    layout = [
        [sg.Image(filename="", key="-WEBCAM-"), 
         sg.Column([[sg.Text("", size=(25,1), font=("Helvetica", 20), justification="left", key="-RESULTS1-")],
                    [sg.Text("", size=(25,1), font=("Helvetica", 20), justification="left", key="-RESULTS2-")],
                    [sg.Text("", size=(25,1), font=("Helvetica", 20), justification="left", key="-RESULTS3-")],
                    [sg.Text("", size=(25,1), font=("Helvetica", 20), justification="left", key="-RESULTS4-")],
                    [sg.Text("", size=(25,10), font=("Helvetica", 20), justification="left", key="-RESULTS5-")]])]
    ]

    window = sg.Window("PPE DETECTION", layout, resizable=True, finalize=True)
    guicount = 0

    # Dataloader
    bs = 1
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        for i, frame in enumerate(im0s):  # Loop over frames
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0 = path[i], im0s[i].copy()
                annotator = Annotator(frame.copy(), line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                    
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Webcam GUI
                im0 = annotator.result()
                im0 = cv2.resize(im0, (800,400))
                imgbytes = cv2.imencode('.png', im0)[1].tobytes()
                window["-WEBCAM-"].update(data=imgbytes)
                results = f"{s}{'' if len(det) else ['No Detections', 'No Detections']}"

                #Default
                RED.start(100)
                GREEN.start(0)
                BLUE.start(0)
                pwm.start(0)
                buzzer.off()

                #Results
                results_array = results.strip().split(',')
                print(results_array)

                #Results assigning
                #Checking Gloves
                if results_array[0] == '2 Gloves' or results_array[1] == ' 2 Gloves':
                    results1 = '2 Gloves'
                elif results_array[0] == '1 Glove' or results_array[1] == ' 1 Glove':
                    results1 = 'Incomplete Gloves'
                else:
                    results1 = 'No Gloves'

                #Checking Mask
                if results_array[1] == ' 1 Mask' or results_array[0] == '1 Mask':
                    results2 = '1 Mask'
                else:
                    results2 = 'No Mask'

                #Add GUI counter
                results3 = 'Please wear complete PPE'

                if results_array[0] == '2 Gloves' and results_array[1] == ' 1 Mask':
                #if '1 No Mask' in results_array: #for testing only
                    guicount += 1  # Increment guicount when condition is met
                    if guicount > 0 and guicount <= 10:
                        results3 = 'Please hold for 3'
                        RED.start(0)
                        GREEN.start(0)
                        BLUE.start(100)
                    elif guicount > 10 and guicount <= 20:
                        results3 = 'Please hold for 2'
                        RED.start(0)
                        GREEN.start(0)
                        BLUE.start(100)
                    elif guicount > 20 and guicount <= 30:
                        results3 = 'Please hold for 1'
                        RED.start(0)
                        GREEN.start(0)
                        BLUE.start(100)
                    else:
                        results3 = 'ACCESS GRANTED'
                        RED.start(0)
                        GREEN.start(100)
                        BLUE.start(0)
                        pwm.start(50)
                        buzzer.on()
                        time.sleep(3)
                                        
                else:
                    guicount = 0
                    RED.start(100)
                    GREEN.start(0)
                    BLUE.start(0)

                #update GUI texts
                window["-RESULTS1-"].update(results1)
                window["-RESULTS2-"].update(results2)
                window["-RESULTS3-"].update(results3)
                #window["-RESULTS4-"].update(results4)
                #window["-RESULTS5-"].update(results5)
                # GUI event handling
                event, values = window.read(timeout=20)
                if event == sg.WINDOW_CLOSED:
                    window.close()
                    return
                break

def main():
    run()

if __name__ == "__main__":
    main()