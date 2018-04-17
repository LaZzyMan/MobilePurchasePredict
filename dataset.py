import csv
from pymongo import MongoClient
import re


def generate_train_set():
    db = MongoClient().Surpass
    user_coll = db.user_final_1
    user_predict_coll = db.user_predict
    statistic_coll = db.statistic_final_1
    rate_dict = statistic_coll.find_one({'id': 'top5Dic'})['top5Dic']
    top5_dict = statistic_coll.find_one({'id': 'indexDic'})['indexDic']
    major_dict = statistic_coll.find_one({'id': 'major'})['major']
    name_labels = ['购买', 'id', '年龄', '专业', '性别', '学历', '平均停留时间']
    website_dict = statistic_coll.find_one({'id': 'websiteDic'})['websiteDic']
    shopping_dict = statistic_coll.find_one({'id': 'shopping'})['shopping']
    video_dict = statistic_coll.find_one({'id': 'video'})['video']
    index_labels = ['next_step', 'id', 'age', 'major', 'gender', 'edu', 'stay_time']
    for name, index in top5_dict.items():
        name_labels.append(name)
        index_labels.append(index)
    for name, index in rate_dict.items():
        name_labels.append(name)
        index_labels.append(index)
    for name, index in website_dict.items():
        name_labels.append(name)
        index_labels.append(index)
    for name, index in shopping_dict.items():
        name_labels.append(name)
        index_labels.append(index)
    for name, index in video_dict.items():
        name_labels.append(name)
        index_labels.append(index)
    with open('user_2_train.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(index_labels[: 7] + index_labels[41:])
        writer.writerow(name_labels[: 7] + name_labels[41:])
        users = user_coll.find({})
        for user in users:
            data_line = [-1 for i in range(len(name_labels))]
            # data_line[6: len(top5_dict) + 6] = ['f' for i in range(len(top5_dict))]
            if user[index_labels[0]] != 'buy':
                data_line[0] = 0
            else:
                data_line[0] = 1
            data_line[1] = user[index_labels[1]]
            data_line[6] = user[index_labels[6]]
            if 'basic_labels' in user.keys():
                data_line[2] = user['basic_labels'][index_labels[2]]
                data_line[3] = int(re.sub('major', '', user['basic_labels'][index_labels[3]]))
                if user['basic_labels'][index_labels[4]] == 'male':
                    data_line[4] = 0
                else:
                    data_line[4] = 1
                data_line[5] = user['basic_labels'][index_labels[5]]
            if 'behavior_labels' in user.keys():
                for i in range(7, len(top5_dict) + 7):
                    if index_labels[i] in user['behavior_labels'].keys():
                        data_line[i] = int(float(user['behavior_labels'][index_labels[i]]))
                for i in range(len(top5_dict) + 7, len(rate_dict) + len(top5_dict) + 7):
                    if index_labels[i] in user['behavior_labels'].keys():
                        data_line[i] = 1
                    else:
                        data_line[i] = 0
            if 'website_labels' in user.keys():
                for i in range(len(top5_dict) + len(rate_dict) + 7,
                               len(website_dict) + len(top5_dict) + len(rate_dict) + 7):
                    if index_labels[i] in user['website_labels'].keys():
                        data_line[i] = user['website_labels'][index_labels[i]]['name_count']
            if 'shopping_labels' in user.keys():
                for i in range(len(website_dict) + len(top5_dict) + len(rate_dict) + 7,
                               len(shopping_dict) + len(website_dict) + len(top5_dict) + len(rate_dict) + 7):
                    if index_labels[i] in user['shopping_labels'].keys():
                        data_line[i] = int(float(user['shopping_labels'][index_labels[i]]))
            if 'vedio_labels' in user.keys():
                for i in range(len(shopping_dict) + len(website_dict) + len(top5_dict) + len(rate_dict) + 7,
                               len(index_labels)):
                    if index_labels[i] in user['vedio_labels'].keys():
                        data_line[i] = int(float(user['vedio_labels'][index_labels[i]]))
            writer.writerow(data_line[: 7] + data_line[41: ])
        f.close()
        with open('user_2_test.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(index_labels[: 7] + index_labels[41: ])
            users = user_predict_coll.find({})
            for user in users:
                data_line = [-1 for i in range(len(name_labels))]
                # data_line[6: len(top5_dict) + 6] = ['f' for i in range(len(top5_dict))]
                data_line[0] = 0
                data_line[1] = user[index_labels[1]]
                data_line[6] = user[index_labels[6]]
                if 'basic_labels' in user.keys():
                    data_line[2] = user['basic_labels'][index_labels[2]]
                    data_line[3] = int(re.sub('major', '', user['basic_labels'][index_labels[3]]))
                    if user['basic_labels'][index_labels[4]] == 'male':
                        data_line[4] = 0
                    else:
                        data_line[4] = 1
                    data_line[5] = user['basic_labels'][index_labels[5]]
                if 'behavior_labels' in user.keys():
                    for i in range(7, len(top5_dict) + 7):
                        if index_labels[i] in user['behavior_labels'].keys():
                            data_line[i] = int(float(user['behavior_labels'][index_labels[i]]))
                    for i in range(len(top5_dict) + 7, len(rate_dict) + len(top5_dict) + 7):
                        if index_labels[i] in user['behavior_labels'].keys():
                            data_line[i] = 1
                        else:
                            data_line[i] = 0
                if 'website_labels' in user.keys():
                    for i in range(len(top5_dict) + len(rate_dict) + 7,
                                   len(website_dict) + len(top5_dict) + len(rate_dict) + 7):
                        if index_labels[i] in user['website_labels'].keys():
                            data_line[i] = user['website_labels'][index_labels[i]]['name_count']
                if 'shopping_labels' in user.keys():
                    for i in range(len(website_dict) + len(top5_dict) + len(rate_dict) + 7,
                                   len(shopping_dict) + len(website_dict) + len(top5_dict) + len(rate_dict) + 7):
                        if index_labels[i] in user['shopping_labels'].keys():
                            data_line[i] = int(float(user['shopping_labels'][index_labels[i]]))
                if 'vedio_labels' in user.keys():
                    for i in range(len(shopping_dict) + len(website_dict) + len(top5_dict) + len(rate_dict) + 7,
                                   len(index_labels)):
                        if index_labels[i] in user['vedio_labels'].keys():
                            data_line[i] = int(float(user['vedio_labels'][index_labels[i]]))
                writer.writerow(data_line[: 7] + data_line[41: ])
            f.close()


if __name__ == '__main__':
    generate_train_set()




