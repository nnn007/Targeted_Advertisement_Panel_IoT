# Import required modules
import math
import time
from time import sleep
import argparse
import cv2 as cv
from picamera import PiCamera
import boto3
import botocore

BUCKET_NAME = 'dop-ads'
s3 = boto3.resource('s3')

img_path = 'images/test_img.jpg'
ad_path = 'ads/'

camera = PiCamera()
ad_time = 10000    #10 seconds

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')



camera.start_preview()
sleep(5)
camera.capture(img_path)
camera.stop_preview()


img = cv.imread(img_path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print (len(faces))     #no of faces in the image

#define the screen resulation
screen_res = 1280, 720
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
#resized window width and height
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

if(len(faces)==1):

    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn, bboxes


    #parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    #parser.add_argument('--input',
     #                  help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    #args = parser.parse_args()

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    # Open a video file or an image file or a camera stream
    #cap = cv.VideoCapture(args.input if args.input else 0)
    cap = cv.VideoCapture(img_path)
    padding = 20
    while cv.waitKey(1) < 0:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            #print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                   max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
            #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            #print("Age Output : {}".format(agePreds))
            #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                       cv.LINE_AA)
            #cv.imshow("Age Gender Demo", frameFace)
            # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)

            #CATALOGUE OF DIFFERENT ADS
            if(age=='(0-2)'):
                if(gender=='Male'):
                    #print('Ad 1')
                    KEY = 'c1.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'baby_product_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                    
                    ad = cv.imread(ad_path + 'baby_product_ad.jpg')
                    ad_window = 'Baby Product Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 2')
                    KEY = 'c2.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'baby_product_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'baby_product_ad.jpg')
                    ad_window = 'Baby Product Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                    
                    
            elif(age=='(4-6)'):
                if(gender=='Male'):
                    #print('Ad 3')
                    KEY = 'c3.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'little_kids_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'little_kids_ad.jpg')
                    ad_window = 'Little Kids Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 4')
                    KEY = 'c4.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'little_kids_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'little_kids_ad.jpg')
                    ad_window = 'Little Kids Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                    
                    
            elif(age=='(8-12)'):
                if(gender=='Male'):
                    #print('Ad 5')
                    KEY = 'c5.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'school_kids_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'school_kids_ad.jpg')
                    ad_window = 'School Kids Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 6')
                    KEY = 'c6.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'school_kids_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'school_kids_ad.jpg')
                    ad_window = 'School Kids Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                    
            elif(age=='(15-20)'):
                if(gender=='Male'):
                    #print('Ad 7')
                    KEY = 'c7.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'college_kids_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'college_kids_ad.jpg')
                    ad_window = 'College Kids Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 8')
                    KEY = 'c8.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'college_kids_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'college_kids_ad.jpg')
                    ad_window = 'College Kids Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                    
                    
            elif(age=='(25-32)'):
                if(gender=='Male'):
                    #print('Ad 9')
                    KEY = 'c9.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'gillete_men_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'gillete_men_ad.jpg')
                    ad_window = 'Gillete Men Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 10')
                    KEY = 'c10.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'gillete_men_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'nykaa_women_ad.jpg')
                    ad_window = 'Nykaa Women Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                    
                    
            elif(age=='(38-43)'):
                if (gender == 'Male'):
                    #print('Ad 11')
                    KEY = 'c11.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'apple_men_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'apple_men_ad.jpg')
                    ad_window = 'Apple Men Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 12')
                    KEY = 'c12.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'biba_mother_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'biba_mother_ad.jpg')
                    ad_window = 'Biba Mother Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                    
                    
            elif(age=='(48-53)'):
                if (gender == 'Male'):
                    #print('Ad 13')
                    KEY = 'c13.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'property_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'property_ad.jpg')
                    ad_window = 'Property Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 14')
                    KEY = 'c14.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'property_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'property_ad.jpg')
                    ad_window = 'Property Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                    
                    
            elif(age=='(60-100)'):
                if (gender == 'Male'):
                    #print('Ad 15')
                    KEY = 'c15.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'old_age_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'old_age_ad.jpg')
                    ad_window = 'Old Age Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()
                else:
                    #print('Ad 16')
                    KEY = 'c16.jpg'
                    try:
                        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'old_age_ad.jpg')
                    except botocore.exceptions.ClientError as e:
                        if e.response['Error']['Code'] == "404":
                            print("The object does not exist.")
                        else:
                            raise
                            
                    ad = cv.imread(ad_path + 'old_age_ad.jpg')
                    ad_window = 'Old Age Ad'
                    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(ad_window, window_width, window_height)
                    cv.imshow(ad_window, ad)
                    cv.waitKey(ad_time)
                    cv.destroyAllWindows()

            break       #allowing ad for the highest confidence prediction (See test2)
        #print("time : {:.3f}".format(time.time() - t))

else:
    #print("General Ad")
    KEY = 'general.jpg'
    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, ad_path+'mmt_general_ad.jpg')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
        
    ad = cv.imread(ad_path+'mmt_general_ad.jpg')
    ad_window = 'General Ad'
    cv.namedWindow(ad_window, cv.WINDOW_NORMAL)
    cv.resizeWindow(ad_window, window_width, window_height)
    cv.imshow(ad_window, ad)
    cv.waitKey(ad_time)
    cv.destroyAllWindows()




