import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

malesCount = 0
femalesCount = 0

bucket0to2 = 0
bucket4to6 = 0
bucket8to12 = 0
bucket15to20 = 0
bucket25to32 = 0
bucket38to43 = 0
bucket48to53 = 0
bucket60to100 = 0

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20

while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    # print("The Frame is: ", frame.shape)

    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # print("the no of faces detected",len(faceBoxes))
    # print ("The Facebox items:",faceBoxes )
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),
        		   max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        # print("The face in for loop of faceboxes", face)

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        # print("the blob is",blob)

        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        if(gender == 'Female'): 
        	femalesCount = femalesCount + 1
        else:
        	malesCount = malesCount + 1


        if(age == '(0-2)'):
        	bucket0to2 = bucket0to2 + 1
        elif(age == '(4-6)'):
        	bucket4to6 = bucket4to6 + 1
        elif(age == '(8-12)'):
        	bucket8to12 = bucket8to12 + 1
        elif(age == '(15-20)'):
        	bucket15to20 = bucket15to20 + 1
        elif(age == '(25-32)'):
        	bucket25to32 = bucket25to32 + 1
        elif(age == '(38-43)'):
        	bucket38to43 = bucket38to43 + 1
        elif(age == '(48-53)'):
        	bucket48to53 = bucket48to53 + 1
        elif(age == '(60-100)'):
        	bucket60to100 = bucket60to100 + 1


        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

    print("------------------------------------------")
    print("Total Males--",malesCount)
    print("Total Females--",femalesCount)
    print("------------------------------------------")
    if(bucket0to2):
    	print("The no of people in (0-2) age group are:",bucket0to2)
    if(bucket4to6):
    	print("The no of people in (4-6) age group are:",bucket4to6)
    if(bucket8to12):
    	print("The no of people in (8-12) age group are:",bucket8to12)
    if(bucket15to20):
    	print("The no of people in (15-20) age group are:",bucket15to20)
    if(bucket25to32):
    	print("The no of people in (25-32) age group are:",bucket25to32)
    if(bucket38to43):
    	print("The no of people in (38-43) age group are:",bucket38to43)
    if(bucket48to53):
    	print("The no of people in (48-53) age group are:",bucket48to53)
    if(bucket60to100):
    	print("The no of people in (60-100) age group are:",bucket60to100)


