# -*- coding: utf-8 -*-
'''
将事件插入数据库主程序

用法：

'''

import datetime
import argparse
import pymysql #导入模块


connect = pymysql.connect(host='localhost',  # 本地数据库
                                   user='root',
                                   password='jiangshan201310.',
                                   db='old_care',
                                   charset='utf8')  # 服务器名,账户,密码，数据库名称
cursor = connect.cursor()

f = open('allowinsertdatabase.txt','r')
content = f.read()
f.close()
allow = content[11:12]

if allow == '1': # 如果允许插入
# if allow:
    f = open('allowinsertdatabase.txt','w')
    f.write('is_allowed=0')
    f.close()
    
    print('准备插入数据库')
    
    # 传入参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-ed", "--event_desc", required=False, 
                    default = '', help="")
    ap.add_argument("-et", "--event_type", required=False, 
                    default = '', help="")
    ap.add_argument("-el", "--event_location", required=False, 
                    default = '', help="")
    ap.add_argument("-epi", "--old_people_id", required=False, 
                    default = '', help="")
    ap.add_argument("-image","--image",required=False,
                    default='',help="")
    args = vars(ap.parse_args())
    
    event_desc = args['event_desc']
    event_type = int(args['event_type']) if args['event_type'] else None
    event_location = args['event_location']
    old_people_id = int(args['old_people_id']) if args['old_people_id'] else None
    # event_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_date = datetime.datetime.now().strftime("%Y-%m-%d")
    image = args['image']
    print(image)
    payload = {'id': 0,  # id=0 means insert; id=1 means update;
               'event_desc': event_desc,
               'event_type': event_type,
               'event_date': event_date,
               'event_location': event_location,
               'oldperson_id': old_people_id,
               'image': image}
    print(payload)
    print('调用插入事件数据的API')
    print(old_people_id)
    if old_people_id==None:
        print("asjdi")
        old_people_id=0
    if event_type == 1:
        event_type = '义工老人交互'
    elif event_type == 3:
        event_type = '老人摔倒'
    elif event_type == 4:
        event_type = '进入禁止区域'
    elif event_type == 0:
        event_type = '老人微笑'
    elif event_type == 2:
        event_type = '出现陌生人'
    add = "INSERT INTO event_info (event_type, event_date, event_location, event_desc, oldperson_id, url) values('{}','{}','{}','{}','{}','{}')".format(event_type,event_date, event_location, event_desc, old_people_id, image)
    cursor.execute(add)
    connect.commit()

    if(event_location == '房间桌子'):
        add1 = "insert into desk_info (event_type,image) values ('{}','{}')".format(event_type, image)
        cursor.execute(add1)
        connect.commit()
    elif(event_location == '走廊'):
        print(123)
        insert2 = "insert into corridor_info (event_type,image) values ('{}','{}')".format(event_type, image)
        cursor.execute(insert2)
        connect.commit()
    elif(event_location == '院子'):
        insert3 = "insert into yard_info (event_type, image) values ('{}','{}')".format(event_type, image)
        cursor.execute(insert3)
        connect.commit()
    else:
        insert4 = "insert into room_info (event_type,image) values ('{}','{}')".format(event_type, image)
        cursor.execute(insert4)
        connect.commit()

    
    print('插入成功')
    
else:
    print('just pass')

