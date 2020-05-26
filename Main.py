import cv2
import numpy as np
import threading
import time
from imgurpython import ImgurClient
from threading import Lock
from twilio.rest import Client

DEBUG = False
#-----------------------------------------------------------------------------------------

def uploadImg(path):
    clientId = "f69072d11c8640d" # Not secure
    clientSecret = "4ca7e6f64b54475017869a9f791da43524293240" # Not secure

    client = ImgurClient(clientId, clientSecret)

    img = client.upload_from_path(path)

    return img["link"]

#-----------------------------------------------------------------------------------------

ACCOUNT_SID = "ACf401bcd3172987e1c9b06e0ed92d7bf5" # Not secure
AUTH_TOKEN = "f0c3fbce7200038bc188dbc5dfa1114a" # Not secure
client = Client(ACCOUNT_SID, AUTH_TOKEN)

def sendSMS(body, url):
    global DEBUG
    if not DEBUG:
        message = client.messages.create(body=body, from_="+12058318494", to="+15124061231", media_url=url)
    else:
        message = False
        print(body)
    return message

#-----------------------------------------------------------------------------------------

rtspLink = "rtsp://admin:9401@192.168.10.200:554/cam/realmonitor?channel=3&subtype=0"

source = cv2.VideoCapture(rtspLink)

latestFrame = None
lastRet = None
lo = Lock()

def rtspCamBuffer(vcap):
    global latestFrame, lo, lastRet
    while True:
        with lo:
            lastRet, latestFrame = vcap.read()

t1 = threading.Thread(target=rtspCamBuffer,args=(source,),name="rtspReadThread")
t1.daemon=True
t1.start()

#-----------------------------------------------------------------------------------------

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = ["person"]

layer_names = net.getLayerNames()
outputLayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#-----------------------------------------------------------------------------------------

INTERVAL = 120
lastPositive = 0
count = 0

while True:
    frame = None
    if(lastRet is not None) and (latestFrame is not None):
        frame = latestFrame.copy()
    else:
        print("unable to read the frame")
        time.sleep(0.2)
        continue

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame,0.00392, (224,224), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputLayers)

    personFound = False
    personConfidence = 0
    personX = 0
    personY = 0
    personWidth = 0
    personHeight = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence > 0.55) and (class_id == 0):
                personFound = True
                personConfidence = confidence
                personX = int(detection[0] * width)
                personY = int(detection[1] * height)
                personWidth = int(detection[2]*width)
                personHeight = int(detection[3]*height)
                break
        if personFound:
            break

    pt1 = (int(personX-(personWidth/2)),int(personY+(personHeight/2)))
    pt2 = (int(personX+(personWidth/2)),int(personY-(personHeight/2)))

    textPos = (int(personX-(personWidth/2)),int(personY-(personHeight/2)-5))

    cv2.rectangle(frame, pt1, pt2, (0,255,0), 2)
    cv2.putText(frame,"Person: %d%%" % (personConfidence*100),textPos,cv2.FONT_HERSHEY_TRIPLEX,0.7,(255,255,255),1)

    if time.time() - lastPositive >= INTERVAL and personFound:
        cv2.imwrite("Images/detected%d.png" % count, frame)
        url = uploadImg("Images/detected0.png")
        sendSMS("Alert: Person Detected!", url)
        lastPositive = time.time()
        count += 1

    frame = cv2.resize(frame,(500,500))
    cv2.imshow("Stream", frame)
    cv2.waitKey(1)

source.release()
cv2.destroyAllWindows()
