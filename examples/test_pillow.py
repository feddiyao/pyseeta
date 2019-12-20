# MIT License

# Copyright (c) 2017 Tuxedo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import datetime
import os

from datebase.open_person import OPENPERSON
from datebase.open_property import OPENPROPERTY
from pyseeta import Detector
from pyseeta import Aligner
from pyseeta import Identifier
import cv2
import time
import numpy
import face_recognition
from PIL import ImageDraw, ImageFont, Image
from face_recognition import load_image_file
from cv2 import resize


try:
    from PIL import Image, ImageDraw
    import numpy as np
except ImportError:
    raise ImportError('Pillow can not be found!')

vediopath = "/Users/zhaxiaohui/Downloads/工程实践/ch09_20190518235959.mp4"
vediodir = '/Users/zhaxiaohui/Downloads/工程实践/vedio'


def test_detector(frame):
    # load model
    detector = Detector(model_path='/Users/zhaxiaohui/pyseeta/model/seeta_fd_frontal_v1.0.bin')
    detector.set_min_face_size(30)

    image_color = Image.fromarray(frame).convert('RGB')
    image_gray = image_color.convert('L')
    faces = detector.detect(image_gray)
    draw = ImageDraw.Draw(image_color)

    return faces
    # image_color.show()
    # detector.release()


def test_aligner():
    print('test aligner:')
    # load model
    detector = Detector()
    detector.set_min_face_size(30)
    aligner = Aligner()

    image_color = Image.open('data/chloecalmon.png').convert('RGB')
    image_gray = image_color.convert('L')
    faces = detector.detect(image_gray)
    draw = ImageDraw.Draw(image_color)
    draw.ellipse ((0,0,40,80), fill=128)
    for face in faces:
        landmarks = aligner.align(image_gray, face)
        for point in landmarks:
            x1, y1 = point[0] - 2, point[1] - 2
            x2, y2 = point[0] + 2, point[1] + 2
            draw.ellipse((x1,y1,x2,y2), fill='red')
    image_color.show()

    aligner.release()
    detector.release()


def test_identifier():
    print('test identifier:')
    detector = Detector()
    aligner = Aligner()
    identifier = Identifier()

    # load image
    image_color_A = Image.open('data/single.jpg').convert('RGB')
    image_gray_A = image_color_A.convert('L')
    image_color_B = Image.open('data/double.jpg').convert('RGB')
    image_gray_B = image_color_B.convert('L')

    # detect face in image
    faces_A = detector.detect(image_gray_A)
    faces_B = detector.detect(image_gray_B)

    draw_A = ImageDraw.Draw(image_color_A)
    draw_B = ImageDraw.Draw(image_color_B)

    if len(faces_A) and len(faces_B):
        landmarks_A = aligner.align(image_gray_A, faces_A[0])
        featA = identifier.extract_feature_with_crop(image_color_A, landmarks_A)
        draw_A.rectangle([(faces_A[0].left, faces_A[0].top), (faces_A[0].right, faces_A[0].bottom)], outline='green')

        sim_list = []
        for face in faces_B:
            landmarks_B = aligner.align(image_gray_B, face)
            featB = identifier.extract_feature_with_crop(image_color_B, landmarks_B)
            sim = identifier.calc_similarity(featA, featB)
            sim_list.append(sim)
        print('sim: {}'.format(sim_list))
        index = np.argmax(sim_list)
        for i, face in enumerate(faces_B):
            color = 'green' if i == index else 'red'
            draw_B.rectangle([(face.left, face.top), (face.right, face.bottom)], outline=color)

    image_color_A.show()
    image_color_B.show()

    identifier.release()
    aligner.release()
    detector.release()


def test_cropface():
    detector = Detector()
    detector.set_min_face_size(30)
    aligner = Aligner()
    identifier = Identifier()

    image_color = Image.open('data/chloecalmon.png').convert('RGB')
    image_gray = image_color.convert('L')

    faces = detector.detect(image_gray)
    for face in faces:
        landmarks = aligner.align(image_gray, face)
        crop_face = identifier.crop_face(image_color, landmarks)
        Image.fromarray(crop_face).show()

    identifier.release()
    aligner.release()
    detector.release()


def absdiff_demo(image_1, image_2, sThre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)  # 灰度化

    gray_image_1 = cv2.GaussianBlur(gray_image_1, (5, 5), 0)  # 高斯滤波

    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    gray_image_2 = cv2.GaussianBlur(gray_image_2, (5, 5), 0)

    d_frame = cv2.absdiff(gray_image_1, gray_image_2)

    ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY)

    return d_frame


# 人脸特征编码集合
known_face_encodings_new = []

# 人脸特征姓名集合
known_face_IDs_new = []


def readtime(vediodir):
    for filename in os.listdir(vediodir):
        file_name = filename.split('.')[0]
        datestring = file_name.split('_')[1]
        list_i = list(datestring)    # str -> list
        list_i.insert(4, '-')   # 注意不用重新赋值
        list_i.insert(7, '-')
        list_i.insert(10, ' ')  # 注意不用重新赋值
        list_i.insert(13, ':')
        list_i.insert(16, ':')

        str_i = ''.join(list_i)    # list -> str
        # print(str_i)
        date_time = datetime.datetime.strptime(str_i, "%Y-%m-%d %H:%M:%S")
        # print(date_time)
    return date_time


def compareface(image, region, name):
    openperson = OPENPERSON()
    flag = 1
    pic_image = image.crop(region)
    # face_locations = face_recognition.face_locations(pic_image)
    frame_4 = cv2.cvtColor(numpy.asarray(pic_image), cv2.COLOR_RGB2BGR)
    frame_4 = resize(frame_4, (100, 100))
    pic_encodings = face_recognition.face_encodings(frame_4)
    known_face_encodings_new, known_face_IDs_new = openperson.select_encoding()
    if len(pic_encodings) == 0:
        ID = 0
        flag = 3
        return flag, ID
    pic_encoding = pic_encodings[0]
    face_distances = face_recognition.face_distance(known_face_encodings_new, pic_encoding)
    face_distances_list = face_distances.tolist()
    if len(face_distances_list) == 0:
        known_face_encodings_new.append(pic_encoding)
        known_face_IDs_new.append(name)  # 调用的时候外面的nameid要+1
        pic_encoding_str = openperson.change_encoding(pic_encoding)
        openperson.save_encoding()
        openpropety = OPENPROPERTY(pic_encoding_str, 1, readtime(vediodir))
        openpropety.save_property()
        ID = 0
        flag = 2
        return flag, ID
    print(face_distances_list[:])
    minindex = face_distances_list.index(min(face_distances_list))
    if min(face_distances_list) > 0.45:
        known_face_encodings_new.append(pic_encoding)
        known_face_IDs_new.append(name) # 调用的时候外面的nameid要+1
        pic_encoding_str = openperson.change_encoding(pic_encoding)
        openperson.save_encoding(pic_encoding_str)
        openpropety = OPENPROPERTY(pic_encoding_str, 1, readtime(vediodir))
        openpropety.save_property()
        ID = 0
        flag = 0
    else:
        ID = known_face_IDs_new[minindex]
        pic_encoding_str = openperson.change_encoding(pic_encoding)
        openpropety = OPENPROPERTY(pic_encoding_str, ID, readtime(vediodir))
        openpropety.save_property()
    return flag, ID


def read_vedio():
    openperson = OPENPERSON()
    video_capture = cv2.VideoCapture(vediopath)
    sThre = 10  # sThre表示像素阈值
    i = 0
    countsnap=0

    name = 0

    while True:

        ret, frame = video_capture.read()

        if i == 0:
            cv2.waitKey(1)

            i = i + 1

        ret_2, frame_2 = video_capture.read()

        start = time.time()

        segMap = absdiff_demo(frame, frame_2, sThre)

        kernel = np.ones((4, 4), np.uint8)

        segMap = cv2.morphologyEx(segMap, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((20, 20), np.uint8)

        segMap = cv2.morphologyEx(segMap, cv2.MORPH_CLOSE, kernel)

        # binary, contours, hierarchy = cv2.findContours(segMap, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  # 该函数计算一幅图像中目标的轮廓

        contours, hierarchy = cv2.findContours(segMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0 and len(contours) < 10 and countsnap%4==0:

            for c in contours:

                (x, y, w, h) = cv2.boundingRect(c)

                if w * h < 400:
                    continue

                image = Image.fromarray(cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB))

                region = (x, y, x+w, y+h)
                newImage = image.crop(region)

                frame_3 = cv2.cvtColor(numpy.asarray(newImage),cv2.COLOR_RGB2BGR)

                face_locations = []
                face_names = []
                process_this_frame = True

                if process_this_frame:
                    face_locations = test_detector(frame_3)

                process_this_frame = not process_this_frame

                for face_location in face_locations:
                    top = face_location.top + y
                    right = x + face_location.right
                    bottom = y + face_location.bottom
                    left = face_location.left + x
                    region_next = (left,top,right,bottom)
                    flag, id = compareface(image,region_next,name)
                    cv2.rectangle(frame_2, (left, top), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    if flag == 1:
                        # cv2.putText(frame_2, id, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        print(id)
                        print(flag)
                        print('=-=-=-=-=-=-=-=-=-=-=')
                    elif flag == 0:
                        # cv2.putText(frame_2, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        print(name)
                        print(flag)
                        print('=-=-=-=-=-=-=-=-=-=-=')
                        name+=1

                    # cv2.rectangle(frame_2, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)

                    # cv2.putText(frame, '', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame_2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # test_aligner()
        # test_identifier()
        # test_cropface()

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_vedio()
