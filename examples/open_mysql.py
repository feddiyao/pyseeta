import time

import cv2
import face_recognition
import pymysql
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from face_recognition import load_image_file
from spyder.widgets.findinfiles import FILE_PATH
FILE_PATH1 = "./pic/obama.jpg"
FILE_PATH2 = "./pic/like.jpg"
# 人脸特征编码集合
known_face_encodings = []

# 人脸特征姓名集合
known_face_IDs = []


#def load_image(FILE_PATH, ID):
def change_encoding(pic_encoding):
    # pic_image = face_recognition.load_image_file(FILE_PATH)
    # pic_encoding = face_recognition.face_encodings(pic_image)[0]
    encoding__array_list = pic_encoding.tolist()
# 将列表里的元素转化为字符串
    encoding_str_list = [str(i) for i in encoding__array_list]

    # 拼接列表里的字符串
    encoding_str = ','.join(encoding_str_list)

    # 被识别者姓名
   # ID = ID

    # 将人脸特征编码存进数据库
    save_encoding(encoding_str)
    #save_encoding(encoding__array_list)


def load_image1(FILE_PATH):
    pic_image = face_recognition.load_image_file(FILE_PATH)
    pic_encoding = face_recognition.face_encodings(pic_image)[0]
    encoding__array_list = pic_encoding.tolist()
# 将列表里的元素转化为字符串
    encoding_str_list = [str(i) for i in encoding__array_list]

    # 拼接列表里的字符串
    encoding_str = ','.join(encoding_str_list)

    # 被识别者姓名
   # ID = ID

    # 将人脸特征编码存进数据库
    save_encoding(encoding_str)
    #save_encoding(encoding__array_list)


def select_encoding():
    # 创建数据库连接对象
    conn = pymysql.connect('localhost', 'root', '12345678', 'indentity')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = conn.cursor()

    # SQL查询语句
    sql = "select * from person_table"
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        # 返回的结果集为元组
        for row in results:
            ID = row[0]
            encoding = row[1]
            # print("name=%s,encoding=%s" % (name, encoding))
            # 将字符串转为numpy ndarray类型，即矩阵
            # 转换成一个list
            dlist = encoding.strip(' ').split(',')
            # 将list中str转换为float
            dfloat = list(map(float, dlist))
            arr = np.array(dfloat)

            # 将从数据库获取出来的信息追加到集合中
            known_face_encodings.append(arr)
            known_face_IDs.append(ID)

    except Exception as e:
        print(e)

        # 关闭数据库连接
        conn.close()


def compare_faces1(FILE_PATH):
    pic_image = face_recognition.load_image_file(FILE_PATH)
    pic_encoding = face_recognition.face_encodings(pic_image)[0]
    select_encoding()
                # matches：一个返回值为True或者False值的列表，该表指示了known_face_encodings列表的每个成员的匹配结果
                # tolerance：越小对比越严格，官方说法是0.6为典型的最佳值，也是默认值
                # 这里我设置0.45为最佳，可能跟我硬件有关
    # matches = face_recognition.compare_faces(known_face_encodings, pic_encoding, tolerance=0.6)
    face_distances = face_recognition.face_distance(known_face_encodings, pic_encoding)
                # 默认为unknown
                # name = "Unknow"

    face_distances_list = face_distances.tolist()
    minindex = face_distances_list.index(min(face_distances_list))
    if min(face_distances_list) > 0.6:
        print('不存在')
        change_encoding(pic_encoding)
    else:
        ID = known_face_IDs[minindex]
        print('已存在')
        print(ID)


def compare_faces(pic_encoding):
    # pic_image = face_recognition.load_image_file(FILE_PATH)
    # pic_encoding = face_recognition.face_encodings(pic_image)[0]
    select_encoding()
                # matches：一个返回值为True或者False值的列表，该表指示了known_face_encodings列表的每个成员的匹配结果
                # tolerance：越小对比越严格，官方说法是0.6为典型的最佳值，也是默认值
                # 这里我设置0.45为最佳，可能跟我硬件有关
    # matches = face_recognition.compare_faces(known_face_encodings, pic_encoding, tolerance=0.6)
    face_distances = face_recognition.face_distance(known_face_encodings, pic_encoding)
    face_distances_list = face_distances.tolist()
    minindex = face_distances_list.index(min(face_distances_list))
    if min(face_distances_list) > 0.6:
        print('不存在')
        change_encoding(pic_encoding)
    else:
        ID = known_face_IDs[minindex]
        print('已存在')
        print(ID)

def save_encoding(encoding_str):
    # 创建数据库连接对象
    db = pymysql.connect('localhost', 'root', '12345678', 'Indentity')
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()

    # SQL插入语句
    insert_sql = "insert into person_table(face_vector) values(%s)"
    try:
        # 执行sql语句
        cursor.execute(insert_sql, encoding_str)
        # 提交到数据库执行
        db.commit()
    except Exception as e:
        # 如果发生错误则回滚并打印错误信息
        db.rollback()
        print(e)

    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    db.close()


def absdiff_demo(image_1, image_2, sThre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)  # 灰度化

    gray_image_1 = cv2.GaussianBlur(gray_image_1, (5, 5), 0)  # 高斯滤波

    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    gray_image_2 = cv2.GaussianBlur(gray_image_2, (5, 5), 0)

    d_frame = cv2.absdiff(gray_image_1, gray_image_2)

    ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY)

    return d_frame


def load_image():

    # 得到特征信息
    # get_info()

    #  打开摄像头 0代表笔记本的内置摄像头，1代表外置摄像头
    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture('/Users/zhaxiaohui/Downloads/工程实践/ch09_20190518235959.mp4')

    sThre = 10  # sThre表示像素阈值

    i = 0

    process_this_frame = True

    while True:

        ret, frame = video_capture.read()

        if i == 0:
            cv2.waitKey(1)

            i = i + 1

        ret_2, frame_2 = video_capture.read()

        start = time.time()

        segMap = absdiff_demo(frame, frame_2, sThre)

        # cv2.waitKey(1)

        kernel = np.ones((4, 4), np.uint8)

        segMap = cv2.morphologyEx(segMap, cv2.MORPH_OPEN, kernel)

        kernel = np.ones((20, 20), np.uint8)

        segMap = cv2.morphologyEx(segMap, cv2.MORPH_CLOSE, kernel)

        binary, contours, hierarchy = cv2.findContours(segMap, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)  # 该函数计算一幅图像中目标的轮廓

        # contours, hierarchy = cv2.findContours(segMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0 and len(contours) < 10:

            for c in contours:

                (x, y, w, h) = cv2.boundingRect(c)

                if w * h < 400:
                    continue

                image = Image.fromarray(cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB))

                region = (x, y, x+w, y+h)
                newImage = image.crop(region)

                frame_3 = cv2.cvtColor(np.asanyarray(newImage), cv2.COLOR_RGB2BGR)

                # 利用opencv的缩放函数改变摄像头图像的大小，图像越小，所做的计算就少
                small_frame = cv2.resize(frame_3, (0, 0), fx=0.25, fy=0.25)

                # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
                rgb_small_frame = small_frame[:, :, ::-1]

                # 处理每一帧的图像
                if process_this_frame:
                    # 使用默认的HOG模型查找图像中的所有人脸
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    # 如果硬件允许，可以使用GPU进行加速，此时应改为CNN模型
                    # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

                    # 返回128维人脸编码，即人脸特征
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []

                    # 将得到的人脸特征与数据库中的人脸特征集合进行比较，相同返回True，不同返回False
                    for face_encoding in face_encodings:
                        compare_faces(face_encoding)
                        # # matches：一个返回值为True或者False值的列表，该表指示了known_face_encodings列表的每个成员的匹配结果
                        # # tolerance：越小对比越严格，官方说法是0.6为典型的最佳值，也是默认值
                        # # 这里我设置0.45为最佳，可能跟我硬件有关
                        # matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
                        # # 默认为unknown
                        # name = "Unknow"
                        #
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     ID = known_face_IDs[first_match_index]
                        #     print('已存在')
                        # else:
                        #     print('不存在')
                        #     change_encoding(face_encoding)


                        #face_names.append(ID)

                process_this_frame = not process_this_frame

                # # 将捕捉到的人脸显示出来
                # for (top, right, bottom, left), name in zip(face_locations, face_names):
                #     # 恢复显示的图像大小
                #     top *= 4
                #     right *= 4
                #     bottom *= 4
                #     left *= 4
                #
                #     # CV库有自己的编码规范，要想在图像上输出中文，需将图片格式转化为PIL库的格式，用PIL的方法写入中文，然后在转化为CV的格式
                #     # cv2和PIL中颜色的hex码的储存顺序不同
                #     cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     pilimg = Image.fromarray(cv2img)
                #
                #     # PIL图片上打印汉字
                #     draw = ImageDraw.Draw(pilimg)
                #     # NotoSansCJK-Light.ttc为本机上已有的字体，可通过locate *.ttc进行查询
                #     font = ImageFont.truetype("NotoSansCJK-Light.ttc", 30, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
                #     draw.text((left + 10, top - 40), name, (255, 255, 255), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
                #
                #     # PIL图片转cv2 图片
                #     frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
                #
                #     # 对人脸画出矩形框
                #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                #
                #     # 如果只想输出英文，可以省略以上步骤，编写以下代码即可
                #     # 显示的字体类型
                #     # font = cv2.FONT_HERSHEY_TRIPLEX
                #     # 打印识别信息
                #     # cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
                #
                # # 显示图像
                # cv2.imshow('monitor', frame)
                #
                # # 按Q退出
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                for top, right, bottom, left in face_locations:
                    top = top * 4 + y
                    right = right * 4 + x
                    bottom = bottom * 4 + y
                    left = left * 4 + x

                    cv2.rectangle(frame_2, (left, top), (right, bottom), (0, 0, 255),  2)

                    #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    #cv2.putText(frame, '',  (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame_2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头资源
    video_capture.release()
    # 关闭显示图像的窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    load_image1(FILE_PATH1)
    load_image1(FILE_PATH2)
    compare_faces1(FILE_PATH2)
    # load_image()

