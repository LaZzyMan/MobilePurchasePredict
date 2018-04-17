from pymongo import MongoClient
import csv
import re
import jieba

BRAND = ['Surpass', '红米', '黑莓', '大显', '大神', 'Apple', '苹果', '荣耀', '乐视', '华为', '微软', '三星', '小米', 'vivo',
         'OPPO', '魅族', '360', 'HTC', '联想', '努比亚', '中兴', '诺基亚', '金立', '索尼', '摩托罗拉', '纽曼', '飞利浦', '酷派',
         '天语', '长虹', '索爱', '守护宝', '海信', '美图', '优思朵唯', 'LG', '波导', '誉品', '康佳', '锤子', 'TCL', '欧奇', '尼凯',
         '恩海尔', '8848', 'VETAS', 'ZUK', '一加', '金圣达', '虫子', '小辣椒', '奇酷', '华硕', '海尔', '赛博宇华', '魅蓝', '谷歌',
         '朵唯', '爱我', '为美']
WEBSITE = ['京东', '1号店', '易迅网', '苏宁易购']


def loadUserInfo(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            id = row[0]
            time_s = row[1].split(':')
            m_time = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            result_s = row[2]
            result = {'搜索': 2, '浏览': 1, '购买': 0}[result_s]
            coll.insert_one({'id': id, 'm_time': m_time, 'result': result})
            print('loadUserInfo ' + str(count))
        print('loadUserInfo finished')
        f.close()


def loadBasicLabels(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            id = row[1]
            age = {'18-24': 0, '25-34': 1, '35-44': 2, '45-54': 3}[row[2]]
            gender = {'男': 0, '女': 1}[row[3]]
            edu = {'高中及以下': 0, '大学专科': 1, '大学本科': 2, '硕士及以上': 3, '其它': -1}[row[4]]
            profession = row[5].split('_')
            coll.update({'id': id}, {'$set': {'basic_labels': {'age': age, 'gender': gender, 'edu': edu, 'profession': profession}}})
            count += 1
            print('loadBasicLabels ' + str(count))
        print('loadBasicLabels finished')
        f.close()


def loadBasicLabels_1(filename, coll, coll_2):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        majorcount = 0
        major = coll_2.find_one({'id': 'major'})['major']
        reColl={}
        for row in list(reader)[1:]:
            id = row[1]
            age = {'18-24': 0, '25-34': 1, '35-44': 2, '45-54': 3}[row[2]]
            gender = {'男': 'male', '女': 'female'}[row[3]]
            edu = {'高中及以下': 0, '大学专科': 1, '大学本科': 2, '硕士及以上': 3, '其它': -1}[row[4]]
            count += 1
            if row[5] not in major:
                majorcount += 1
                major[row[5]] = 'major' + str(majorcount)
            profession = major[row[5]]
            reColl[id] = {'age': age, 'gender': gender, 'edu': edu, 'major': profession}
            print('loadBasicLabels ' + str(count))
        print('loadBasicLabels finished')
        f.close()
        for id, contents in reColl.items():
            coll.update_one({'id': id}, {'$set': {'basic_labels': contents}})


def loadBehaviorLabels(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            id = row[1]
            behavior_s = re.split(r':', row[2])
            dist = {}
            for i in range(len(behavior_s) - 1):
                key = re.split(r'[,、，\s]', behavior_s[i])[-1]
                if 'top5' in key:
                    if i == len(behavior_s) - 2:
                        value = re.split(r'[,、，\s]', behavior_s[i + 1])
                    else:
                        value = re.split(r'[,、，\s]', behavior_s[i + 1])[0: -1]
                else:
                    value = re.split(r'[,、，\s]', behavior_s[i + 1])[0]
                dist[key] = value
            coll.update({'id': id}, {'$set': {'behavior_labels': dist}})
            count += 1
            print('loadBehaviorLabels ' + str(count))
        print('loadBehaviorLabels finished')
        f.close()


def loadBehaviorLabels_1(filename, coll, coll2):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        reColl = {}
        indexDic = coll2.find_one({'id': 'indexDic'})['indexDic']
        indexNumber = 0
        top5Dic = coll2.find_one({'id': 'top5Dic'})['top5Dic']
        top5Number = 0
        for row in list(reader)[1:]:
            id = row[1]
            behavior_s = re.split(r':', row[2])
            dist = {}
            for i in range(len(behavior_s) - 1):
                key = re.split(r'[,、，\s]', behavior_s[i])[-1]
                if 'top5' in key:
                    if key not in top5Dic:
                        continue
                    if i == len(behavior_s) - 2:
                        value = re.split(r'[,、，\s]', behavior_s[i + 1])
                    else:
                        value = re.split(r'[,、，\s]', behavior_s[i + 1])[0: -1]
                    dist[top5Dic[key]] = value
                    print(top5Dic[key])
                else:
                    if key not in indexDic:
                        continue
                    value = re.split(r'[,、，\s]', behavior_s[i + 1])[0]
                    dist[indexDic[key]] = value
                    print(indexDic[key])
            reColl[id] = dist
            count += 1
            print('loadBehaviorLabels ' + str(count))
        print('loadBasicLabels finished')
        f.close()
        for id, contents in reColl.items():
            coll.update({'id': id}, {'$set': {'behavior_labels': contents}})


def loadEcommerce(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            record = {}
            id = row[1]
            time_s = row[2].split(':')
            record['stay_time'] = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            name_s = re.sub(r'[(【\w+】)(（\w+）)(\(\w+\))]', '', row[3])
            record['name'] = ''.join(list(jieba.cut(name_s))[0: 2])
            if len(name_s.split('-')) >= 2:
                record['website'] = name_s.split('-')[-1]
            else:
                record['website'] = 'Unknown'
            record['class_1'] = row[4]
            record['class_2'] = row[5]
            user = coll.find_one({'id': id})
            if user is None:
                continue
            if 'Ecommerce_behavior' in user:
                user['Ecommerce_behavior'].append(record)
            else:
                user['Ecommerce_behavior'] = [record]
            coll.update({'id': id}, {'$set': user})
            count += 1
            print('loadEcommerce ' + str(count))
        print('loadEcommerce finished')
        f.close()


def loadVideo(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            if count < 525000:
                continue
            id = row[1]
            time_s = row[2].split(':')
            stay_time = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            name = row[3]
            class_1 = row[4]
            class_2 = row[5]
            user = coll.find_one({'id': id})
            if user is None:
                continue
            if 'video_behavior' in user:
                user['video_behavior'].append({'name': name, 'stay_time': stay_time, 'class_1': class_1, class_2: class_2})
            else:
                user['video_behavior'] = [{'name': name, 'stay_time': stay_time, 'class_1': class_1, class_2: class_2}]
            coll.update({'id': id}, {'$set': user})
            print('loadVideo ' + str(count))
        print('loadVideo finished')
        f.close()


def loadVideo_1(filename, coll):
    result = {}
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            id = row[1]
            time_s = row[2].split(':')
            stay_time = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            name = row[3]
            class_1 = row[4]
            class_2 = row[5]
            if id not in result.keys():
                result[id] = [{'name': name, 'stay_time': stay_time, 'class_1': class_1, class_2: class_2}]
            else:
                result[id].append({'name': name, 'stay_time': stay_time, 'class_1': class_1, class_2: class_2})
            print('loadVideo ' + str(count))
        f.close()
        count = 0
        for id, contents in result.items():
            user = coll.find_one({'id': id})
            if user is None:
                continue
            coll.update({'id': id}, {'$set': {'video_behavior': contents}})
            count += 1
            print('loadVideo ' + str(count))
        print('loadVideo finished')


def loadCatalyst(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            id = row[1]
            name = row[2]
            class_name = row[3]
            url = row[4]
            user = coll.find_one({'id': id})
            if user is None:
                continue
            if 'catalyst_behavior' in user:
                user['catalyst_behavior'].append({'name': name, 'class': class_name, 'url': url})
            else:
                user['catalyst_behavior'] = [{'name': name, 'class': class_name, 'url': url}]
            coll.update({'id': id}, {'$set': user})
            count += 1
            print('loadCatalyst ' + str(count))
        print('loadCatalyst finished')
        f.close()


def loadCatalyst_1(filename, coll, coll2):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        reColl = {}#一个用户的所有 网站列表 key = id, 镶嵌多个字典 是一个用户一个网站的信息 key = name
        websiteNameDic = coll2.find_one({'id': 'websiteDic'})['websiteDic']
        websiteNameCount = 0
        for row in list(reader)[1:]:
            id = row[1]
            name = row[2]
            class_name = row[3]
            url = row[4]
            if name not in websiteNameDic:
                continue
            if id not in reColl:
                reColl[id] = {}
                reColl[id][websiteNameDic[name]] = {'name': websiteNameDic[name], 'class': class_name, 'url': url, 'name_count': 1}
            else:
                if websiteNameDic[name] in reColl[id]:
                    reColl[id][websiteNameDic[name]]['name_count'] += 1
                else:
                    reColl[id][websiteNameDic[name]] = {'name': websiteNameDic[name], 'class': class_name, 'url': url, 'name_count': 1}
            count += 1
            print('loadCatalyst ' + str(count))
        print('loadCatalyst finished')
        f.close()
        for id, contents in reColl.items():
            coll.update({'id': id}, {'$set': {'website_labels': contents}})


def loadPhone(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in list(reader)[1:]:
            coll.insert_one({'name': row[0], 'size': row[1], 'cpu': row[2], 'ram': row[3], 'rom': row[4],
                             'font': row[5], 'rear': row[6], 'network': row[7], 'battery': row[8],
                             'os': row[9], 'material': row[10], 'screen_ratio': row[11], 'sim_card': row[12]})
        f.close()


def keyWord(s):
    if re.search(r'【\w+】', s).group(0) is not None:
        key_1 = re.sub(r'[【】]', '', re.search(r'【\w+】', s).group(0))


def phone_data(filename, coll):
    with open('./data/phone.csv', 'w', encoding='utf-8', newline='') as f1:
        writer = csv.writer(f1)
        writer.writerow(['id', 'time', 'brand'])
        with open(filename, encoding='utf-8') as f2:
            reader = csv.reader(f2)
            for row in list(reader)[1:]:
                if row[5] == '手机':
                    tag = 0
                    for brand in BRAND:
                        if brand in row[3]:
                            writer.writerow([row[1], row[2], brand])
                            tag = 1
                            continue
                    if tag == 0:
                        writer.writerow([row[1], row[2], '小众品牌'])

            f2.close()
        f1.close()


def loadEcommerce_1(filename_1, filename_2, coll_1, coll_2):
    users = {}
    with open(filename_1, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            print('loadEcommerce 1 ' + str(count))
            time_s = row[2].split(':')
            stay_time = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            id = row[1]
            if id not in users.keys():
                users[id] = {}
                if len(row[3].split('-')) >= 2:
                    website = row[3].split('-')[-1]
                    if website in WEBSITE:
                        users[id][row[3].split('-')[-1]] = stay_time
                if row[5] == '手机':
                    continue
                else:
                    users[id][row[5]] = stay_time
            else:
                if len(row[3].split('-')) >= 2:
                    website = row[3].split('-')[-1]
                    if website in WEBSITE:
                        if website in users[id].keys():
                            users[id][website] += stay_time
                        else:
                            users[id][website] = stay_time
                if row[5] == '手机':
                    continue
                else:
                    if row[5] in users[id].keys():
                        users[id][row[5]] += stay_time
                    else:
                        users[id][row[5]] = stay_time
        f.close()
    with open(filename_2, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            print('loadEcommerce 2 ' + str(count))
            id = row[0]
            time_s = row[1].split(':')
            stay_time = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            if id in users.keys():
                users[id] = {row[2]: stay_time}
            else:
                if row[2] in users[id].keys():
                    users[id][row[2]] += stay_time
                else:
                    users[id][row[2]] = stay_time
        f.close()
        count = 0
    for id, contents in users.items():
        user = coll_1.find_one({'id': id})
        if user is None:
            continue
        coll_2.insert_one({'id': id, 'e_behavior': contents})
        count += 1
        print('loadEcommerce 3 ' + str(count))
    print('loadEcommerce finished')


def getWebsite(filename):
    websites = {}
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            if len(row[3].split('-')) >= 2:
                website = row[3].split('-')[-1]
                if website not in websites.keys():
                    websites[website] = 1
                else:
                    websites[website] += 1
        for id, contents in websites.items():
            if contents >= 1000:
                print(id)
        f.close()


def similiarity(s1, s2):
    (target, template) = (s2, s1) if len(s1) > len(s2) else (s1, s2)
    count = 0
    for i in target:
        if i in template:
            count += 1
    return len(target) - count < count


def video_classify(filename):
    data_with_class = {}
    data_without_class = {}
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            if row[5] != '未分类':
                if row[3] not in data_with_class.keys():
                    data_with_class[row[3]] = row[4] + '_' + row[5]
            print('get class ', count)
        f.close()
    with open('new_video.csv', 'w', encoding='utf-8', newline='') as w:
        writer = csv.writer(w)
        with open(filename, encoding='utf-8') as f:
            reader = list(csv.reader(f))
            count = 0
            writer.writerow(reader[0])
            for row in reader[1:]:
                count += 1
                if row[5] == '未分类':
                    tag = 0
                    for key in data_with_class.keys():
                        if similiarity(key, row[3]):
                            writer.writerow([row[0], row[1], row[2], row[3], data_with_class[key]])
                            tag = 1
                            break
                    if tag == 0:
                        writer.writerow([row[0], row[1], row[2], row[3], '未分类'])
                else:
                    writer.writerow([row[0], row[1], row[2], row[3], row[4] + '_' + row[5]])
                print('write line ', count)
            f.close()
        w.close()


def loadVideo_2(filename, coll_1, coll_2):
    ignores = []
    with open('./data/ignore', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            ignores.append(line.strip())
        f.close()
    users = {}
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            print('loadVideo 1 ' + str(count))
            if row[4] in ignores:
                continue
            id = row[1]
            time_s = row[2].split(':')
            stay_time = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            if id not in users.keys():
                users[id] = {row[4]: stay_time}
            else:
                if row[4] not in users[id].keys():
                    users[id][row[4]] = stay_time
                else:
                    users[id][row[4]] += stay_time
        f.close()
    for id, contents in users.items():
        user = coll_1.find_one({'id': id})
        if user is None:
            continue
        coll_2.insert_one({'id': id, 'v_labels': contents})
        count += 1
        print('loadVideo 2 ' + str(count))
    print('loadVideo finished')


def load_pred_set(filename, coll):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        for row in list(reader)[1:]:
            count += 1
            print('load predict ', count)
            id = row[0]
            time_s = row[1].split(':')
            stay_time = int(time_s[0]) * 60 + int(time_s[1].split('.')[0])
            coll.insert_one({'id': id, 'stay_time': stay_time})
        f.close()


def input_shopping(userColl, phoneColltest, statisticColl):
    users = userColl.find({})
    for user in users:
        if 'e_labels' in user:
            userColl.update_one({'id': user['id']}, {'$unset': {'e_labels': ''}})
    users = phoneColltest.find({})
    shopping = statisticColl.find_one({'id': 'shopping'})['shopping']
    shoppingcount = 0
    count_a = 0
    for user in users:
        shopping_labels = {}
        for ashopping in user['e_behavior']:
            if ashopping not in shopping:
                continue
        for athing in user['e_behavior']:
            shopping_labels[shopping[athing]] = user['e_behavior'][athing]
        userColl.update_one({'id': user['id']}, {'$set': {'shopping_labels': shopping_labels}})
        count_a += 1
        print(count_a)


def input_video(userColl, videoColltest, statisticColl):
    users = videoColltest.find({})
    videoDic = statisticColl.find_one({'id': 'video'})['video']
    videocount = 0
    count_b = 0
    for user in users:
        video_List = {}
        for avideo in user['v_labels']:
            if avideo not in videoDic:
                continue
        for athing in user['v_labels']:
            video_List[videoDic[athing]] = user['v_labels'][athing]
        userColl.update_one({'id': user['id']}, {'$set': {'vedio_labels': video_List}})
        count_b += 1
        print(count_b)


if __name__ == '__main__':
    db = MongoClient().Surpass
    userColl = db.User
    phoneColl = db.Phone
    loadUserInfo('./data/user.csv', userColl)
    loadBasicLabels_1('./data/basiclabels.csv', db.user_predict, db.statistic_final_1)
    loadBehaviorLabels_1('./data/behaviorlabels.csv', db.user_predict, db.statistic_final_1)
    loadPhone('./data/phone.csv', userColl)
    loadCatalyst_1('./data/catalyst.csv', db.user_predict, db.statistic_final_1)
    loadEcommerce('./data/Ecommerce.csv', userColl)
    loadVideo_1('./data/video.csv', userColl)
    phone_data('./data/Ecommerce.csv', userColl)
    loadEcommerce_1('./data/Ecommerce.csv', './data/phone.csv', db.user_predict, db.user_phone_predict)
    getWebsite('./data/Ecommerce.csv')
    video_classify('./data/video.csv')
    loadVideo_2('./new_video.csv', db.user_predict, db.user_video_predict)
    load_pred_set('./data/predict.csv', db.user_predict)
    input_shopping(db.user_predict, db.user_phone_predict, db.statistic_final_1)
    input_video(db.user_predict, db.user_video_predict, db.statistic_final_1)


def filterVideo(filename):
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        video_content = []
        videoList = ['暗黑者2', '挑战者联盟', '花木兰']
        for row in list(reader)[1:]:
            video = row[3]
            if '《' in video:
                videoname = video[video.index('《')+1: video.index('》')]
                if videoname not in videoList:
                    videoList.append(videoname)
        f.close()
        print(videoList)

    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in list(reader)[1:]:
            video = row[3]
            for videoitem in videoList:
                if videoitem in video:
                    video_content.append(videoitem)
                    break
                else:
                    video_content.append('no')
                    print(video)
        f.close()
        print(video_content)
