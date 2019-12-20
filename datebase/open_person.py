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


class OPENPERSON():
    def change_encoding(self, pic_encoding):
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
        # self.save_encoding(encoding_str)
        #save_encoding(encoding__array_list)
        return encoding_str

    def select_encoding(self):
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

            return known_face_encodings, known_face_IDs

        except Exception as e:
            print(e)

            # 关闭数据库连接
            conn.close()

    def save_encoding(self,encoding_str):
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



