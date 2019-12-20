import pymysql


class OPENPROPERTY():
    def __init__(self, encoding_str, person_id, datetime):
        self.encoding_str = encoding_str
        self.person_id = person_id
        self.datetime = datetime

    def save_property(self):
        # 创建数据库连接对象
        db = pymysql.connect('localhost', 'root', '12345678', 'Indentity')
        # 使用 cursor() 方法创建一个游标对象 cursor
        cursor = db.cursor()
        # SQL插入语句
        insert_sql = "insert into property_table(face_vector,person_id,origin_time) values(%s,%s,%s)"
        try:
            # 执行sql语句
            values = (self.encoding_str, self.person_id,self.datetime)
            cursor.execute(insert_sql, values)
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


