# PPE-Detection-Using-YoloV9
*** ***
[Implementation of the existing git of YoloV9 by WongKinYiu](https://github.com/WongKinYiu/yolov9/)
*** ***
PPE Detection System that detects facemasks and gloves
</br>
</br>
</br>
This project is made to simulate the entry in an area that requires the person to wear 2 gloves and 1 facemask before granting access
</br>
</br>
Simulated using both a personal computer and using a Raspberry Pi 3b+
</br>
</br>
Used a simple GUI and countdown tracker for confirmation that the criteria was met
</br>
</br>
The GUI shows the camera for object decetion and the text that displays which PPEs you have/are lacking
</br>
</br>
</br>
Model Training with pretrained weights of yolov9-c with epochs of 10 with img size of 640
</br>
</br>
Dataset used is from a combination of images from roboflow
</br>
</br>
</br>
File used is named *FINAL.py*
*** ***
<details>
 <summary>Important Links</summary>
</br>
 
 [Dataset](https://universe.roboflow.com/4d/ppe-4ngvv)
</br>
 Weights: *will be uploaded soon*
</details>
<details>
 <summary>Screenshots on the use of the application on a Raspberry Pi 3b+</summary>
 </br>
 </br>
 </br>
 
 *Incomplete PPE/No PPE*
 ![screenshot3](https://github.com/chardizard3/PPE-Detection-Using-YoloV9/blob/main/3.jpg)
 </br>
 </br>
 *Countdown 3*
 ![screenshot5](https://github.com/chardizard3/PPE-Detection-Using-YoloV9/blob/main/5.jpg)
 </br>
 </br>
 *Countdown 2*
 ![screenshot2](https://github.com/chardizard3/PPE-Detection-Using-YoloV9/blob/main/2.jpg)
 </br>
 </br>
 *Countdown 1*
 ![screenshot4](https://github.com/chardizard3/PPE-Detection-Using-YoloV9/blob/main/4.jpg)
 </br>
 </br>
 *Access Granted*
 ![screenshot1](https://github.com/chardizard3/PPE-Detection-Using-YoloV9/blob/main/1.jpg)
</details>
