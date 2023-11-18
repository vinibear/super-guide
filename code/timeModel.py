import pymysql
import numpy as np


# 将一个列表转化为字典，值初始化为0
def list1dict(lst):
    return {key: 0 for key in lst}



def getSubs(jour):
    # 根据期刊名获取主题列表，不包含重复元素
    con = pymysql.connect(host='172.20.10.14', port=3306, user='roott', passwd='123456', db='test', charset='utf8')
    cur = con.cursor()
    sql = "select Subject from sub_url where Journal='%s'" % (jour)
    cur.execute(sql)
    res = cur.fetchall()
    se = set()
    for row in res:
        se.add(row[0])
    s = list(se)
    cur.close()
    con.close()
    return s



# 期刊周期系数计算-1：计算某期刊2023.1-3审稿周期预测值
# 对于给定的期刊和关键词，计算周期系数
# 首先看在这个期刊包含关键词的论文中，统计每个主题的频率
# *特殊情况：指定期刊中没有包含关键词的论文
def timeCal(cur, jour, key):
    # 获取期刊的主题列表
    sublist = getSubs(jour)
    # 转化为字典，方便后续统计
    subDict = list1dict(sublist)
    sql = "select Key_1,Key_2,Key_3,Key_4,Key_5,Key_6,Key_7,Key_8,Key_9,Key_10,Key_11,Key_12,Key_13," \
          "Key_14,Key_15,Key_16,Key_17,Key_18,Key_19,Key_20,Key_21,Key_22,Key_23," \
          "Subject_1,Subject_2,Subject_3,Subject_4,Subject_5,Subject_6,Subject_7," \
          "Subject_8,Subject_9,Subject_10,Subject_11,Subject_12 from time2 where Journal='%s'" % jour

    # Subject_1对应row[23]
    cur.execute(sql)
    res = cur.fetchall()
    for row in res:
        for i in range(23):  # 0-22
            if row[i] and key in row[i].lower():
                for j in range(23, 35):
                    if row[j]:
                        subDict[row[j]] += 1
                break
    # print(subDict)
    # 计算每个主题包含的论文数，以及每个主题所占比例。不包含0值。[(a,3),(b,2),...]
    valuelist = []
    sum = 0
    for (key, value) in subDict.items():
        if value > 0:
            valuelist.append((key, value))
            sum = sum + value

    # print(sum)
    ratelist = []
    for tup in valuelist:
        rate = float("%.2f" % (tup[1] * 1.0 / sum))
        ratelist.append((tup[0], rate))
    # 根据关键词包含的论文数进行排序，
    value_sorted_list = sorted(valuelist, key=lambda x: x[1], reverse=True)
    rate_sorted_list = sorted(ratelist, key=lambda x: x[1], reverse=True)
    # print("value_sorted_list:",value_sorted_list)
    # print("rate_sorted_list:",rate_sorted_list)
    # i从0.9-0递减，依次看是否有主题占比超过i，选择超过i的主题进行曲线拟合
    # *对于选择哪些主题的曲线进行拟合，可以根据测试结果进行调整
    sublist2 = []
    for i in range(9, -1, -1):
        flag = 0
        for tup in rate_sorted_list:
            if tup[1] * 10 >= i:
                sublist2.append(tup[0])
                flag = 1
        if flag == 1:
            break
    # print("sublist2:",sublist2)
    # 进行拟合，获得拟合结果平均值
    prelist = []
    for sub in sublist2:
        # pre = curveFit(jour, sub)
        pre = getPredict(cur, jour, sub)
        # 有效预测值，0-3年，对超出正常范围的进行修正
        if pre <= 0:
            prelist.append(100)
        elif pre > 1100:
            prelist.append(1100)
        else:
            prelist.append(pre)
    # print(prelist)
    premean = float("%.2f" % np.mean(prelist))  # 平均值
    # print("premean:",premean)
    return premean


# 期刊周期系数计算-2：标准化
def timeStandard(cur, jours, key):
    # 对每一个期刊，使用timeCal获取审稿周期预测值
    avatimeDict = {}
    #timeDict = {'type': 'time_score'}
    timeDict={}
    for j in jours:
        # print("期刊：%s" %j)
        avatimeDict[j] = timeCal(cur, j, key)
    # 计算周期系数，用最小值除以每一个期刊的周期值
    minTime = min(avatimeDict.values())

    for j in jours:
        timeDict[j] = float('%.2f' % (minTime / avatimeDict[j]))
        # timeL.append(float("%.2f" %(minTime/at)))
    # print(avatimeDict)
    # print(timeDict)
    avatimeDict['type'] = 'ava_time'
    return [avatimeDict, timeDict]


# 读取数据库中journal_list表，即每个期刊及其论文总数
def getSumDict(cur, jours):
    # 包含关键词的论文数量
    sumDict = list1dict(jours)
    sumDict['type'] = 'total_paper'

    sql = "select journal,number from journal_list"
    cur.execute(sql)
    res = cur.fetchall()
    for row in res:
        sumDict[row[0]] = row[1]
    # type()
    return sumDict


def getPredict(cur, jour, sub):
    sql = "select prediction from time_predict where journal='%s' and subject='%s'" % (jour, sub)
    cur.execute(sql)
    res = cur.fetchall()
    for row in res:
        x = row[0]
        break
    return x

def keyWordsMatch(cur, keyb, jours):
    # key:要查找的关键词
    sql = "select Key_1,Key_2,Key_3,Key_4,Key_5,Key_6,Key_7,Key_8,Key_9,Key_10,Key_11,Key_12,Key_13," \
          "Key_14,Key_15,Key_16,Key_17,Key_18,Key_19,Key_20,Key_21,Key_22,Key_23,Journal,Subject_1 from time2"
    # Key_1-Key_23:0-22; Journal:23; Subject_1:24
    cur.execute(sql)
    res = cur.fetchall()
    # subjectList=[]
    # 包含关键词的论文数量
    numDict = list1dict(jours)
    #numDict['type'] = 'relate_paper'
    # numDict={TOCS:0,TOS:0,TACO:0,'type':'relate_paper'}
    for row in res:
        # 统计每种期刊包含关键词的论文数
        # 从每行的13个关键词中进行匹配
        if row[24]:
            for i in range(23):  # 0-22
                if row[i] and keyb in row[i].lower():
                    numDict[row[23]] = numDict[row[23]] + 1
                    # subjectList.append([row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[21],row[22]])
                    break
    return numDict

# 输入一个关键词，输出一个列表，对应每个期刊的评分
def main(keyb):
    # 初始化
    con = pymysql.connect(host='172.20.10.14', port=3306, user='roott', passwd='123456', db='test', charset='utf8')
    cur = con.cursor()
    label_dir = {'JACM': 0, 'TAAS': 1, 'TACO': 2, 'TECS': 3, 'TKDD': 4,
                 'TOCHI': 5, 'TODAES': 6, 'TODS': 7, 'TWEB': 8, 'TOG': 9,
                 'TOIS': 10, 'TOIT': 11, 'TOMM': 12, 'TOPLAS': 13, 'TOPS': 14,
                 'TOS': 15, 'TOSEM': 16, 'TOSN': 17, 'TRETS': 18}
    #定义评分向量
    result=[0 for i in range(19)]
    #提取期刊列表
    jours=list(label_dir.keys())
    #统计匹配论文数
    numDict=keyWordsMatch(cur,keyb,jours)
    #print(numDict)
    #剔除匹配论文数为0的期刊
    for (key,value) in numDict.items():
        if value==0:
            jours.remove(key)
    #print(jours)
    #计算审稿周期预测评分
    ds2 = timeStandard(cur, jours, keyb)
    #print(ds2[1])
    #将评分结果填入结果向量
    for (key,value) in ds2[1].items():
        result[label_dir[key]]=value
    #print(result)
    #归一化，计算每个期刊的评分占比
    sum=0
    result_std = [0 for i in range(19)]
    for i in range(19):
        sum=sum+result[i]
    for i in range(19):
        result_std[i]=float('%.5f' %(result[i]/sum))
    #print(result_std)
    cur.close()
    con.close()
    return result_std

if __name__ == '__main__':
    # test()
    # r=curveFit('TOCS','Multiple instruction, multiple data')
    # print(r)
    #finalOPtion('system', 0.5)
    main('database')
