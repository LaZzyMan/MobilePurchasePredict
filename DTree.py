from math import log


class Node:
    def __init__(self, condition=-1, value=None, results=None, tb=None, fb=None):
        '''
        :param condition: 判断条件
        :param value: 返回true的匹配值
        :param results:叶节点结果，非叶节点取None
        :param tb:true子节点
        :param fb:false子节点
        '''
        self.condition = condition
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


def divideset(rows, column, value):
    '''
    根据匹配特征将数据集分类
    :param rows: 数据集
    :param column: 特征索引
    :param value: 匹配特征值
    :return: set1, set2
    '''
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    if isinstance(rows[column], list):
        split_function = lambda row: value in row[column]
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)


def uniquecounts(rows):
    '''
    :param rows:数据集
    :return: 数据中各类结果个数
    '''
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def giniimpurity(rows):
    '''
    :param rows: 数据集
    :return: 基尼不纯度测度
    '''
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


def entropy(rows):
    '''
    :param rows:
    :return: 熵测度
    '''
    log2 = lambda x: log(x) / log(2)
    results = uniquecounts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


def variance(rows):
    '''
    :param rows: 数据集
    :return: 方差测度
    '''
    if len(rows) == 0:
        return 0
    data = [float(row[len(row) - 1]) for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance


def getwidth(tree):
    '''
    :param tree:
    :return: 树广度
    '''
    if tree.tb is None and tree.fb is None:
        return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    '''
    :param tree:
    :return: 树深度
    '''
    if tree.tb == None and tree.fb == None: return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


def buildtree(rows, score_function=entropy):
    '''
    递归生成树
    :param rows: 数据集
    :param score_function: 测度指标
    :return:
    '''
    if len(rows) == 0:
        return Node()
    current_score = score_function(rows)
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1
    for col in range(column_count):
        column_values = {}
        for row in rows:
            if column_values[row[col]] != -1:
                column_values[row[col]] = 1
        for value in column_values.keys():
            (set1, set2) = divideset(rows, col, value)
            p = float(len(set1)) / len(rows)
            gain = current_score - p * score_function(set1) - (1 - p) * score_function(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        true_branch = buildtree(best_sets[0])
        false_branch = buildtree(best_sets[1])
        return Node(condition=best_criteria[0], value=best_criteria[1], tb=true_branch, fb=false_branch)
    else:
        return Node(results=uniquecounts(rows))


def classify(observation, tree):
    '''
    :param observation: 输入数据
    :param tree:
    :return: 分类结果
    '''
    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


def prune(tree, min_gain):
    '''
    决策树剪枝
    :param tree:
    :param min_gain: 最小信息增益
    :return:
    '''
    if tree.tb.results is None:
        prune(tree.tb, min_gain)
    if tree.fb.results is None:
        prune(tree.fb, min_gain)

    if tree.tb.results is not None and tree.fb.results is not None:
        # Build a combined dataset
        tb, fb = [], []
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c

        # Test the reduction in entropy
        delta = entropy(tb + fb) - (entropy(tb) + entropy(fb) / 2)
        if delta < min_gain:
            # Merge the branches
            tree.tb, tree.fb = None, None
            tree.results = uniquecounts(tb + fb)


def mdclassify(self, observation, tree):
    '''
    结果分类（处理缺失值）
    :param observation:
    :param tree:
    :return:决策结果
    '''
    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.col]
        if v is None:
            tr, fr = self.mdclassify(observation, tree.tb), self.mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items():
                result[k] = v * tw
            for k, v in fr.items():
                result[k] = v * fw
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.mdclassify(observation, branch)


if __name__ == '__main__':
    pass
