# USAGE
# python gather_examples.py --input videos/real.mov --output dataset/real --detector face_detector --skip 1
# python gather_examples.py --input videos/fake.mp4 --output dataset/fake --detector face_detector --skip 4

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
# 解析了命令行参数
ap = argparse.ArgumentParser()
# 输入视频文件的路径
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video")
# 输出目录的路径，截取的每一张面部图像都存储在这个目录中
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
# 面部检测器的路径。我们将使用 OpenCV 的深度学习面部检测器
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
# 过滤弱面部检测的最小概率，默认值为 50%
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
# 我们不需要检测和存储每一张图像，因为相邻的帧是相似的。
# 因此我们在检测时会跳过 N 个帧，你可以使用这个参数并更改默认值（16）
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
# 加载 OpenCV 的深度学习面部检测器
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
# 打开视频流，并初始化读取帧的数量和执行循环时保存帧的数量
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# loop over frames from the video file stream
while True:
    # grab the frame from the file
    # 抓取一帧并验证
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # increment the total number of frames read thus far
    # 增加计数器
    read += 1

    # check to see if we should process this frame
    if read % args["skip"] != 0:
        continue

    # grab the frame dimensions and construct a blob from the frame
    # 抓取帧的维度，用于缩放边界框
    (h, w) = frame.shape[:2]
    # 根据图像创建一个 blob。为了适应 Caffe 面部识别器，这个 blob 是 300*300 的
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    # 通过深度学习面部识别器执行了 blob 的前向传输
    net.setInput(blob)
    detections = net.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            # write the frame to disk
            p = os.path.sep.join([args["output"], "NO.2_yellow_fake{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
