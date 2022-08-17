import numpy as np
import time
import random
import math
import copy


class Dijkstra:

    def __init__(self, w, h, img):
        self.w = w
        self.h = h
        self.nnum = img

    # def shuzu(w, h):
    #     nnum = [[random.randint(0, 10) for y in range(h)] for x in range(w)]
    #     nnum = np.array(nnum) / 10
    #     return nnum

    def get_neighbors(self, p):
        x, y = p  # 当前点坐标
        # 3*3领域范围
        #########################
        # 在图像领域 xy轴互换 left改成上（后退），top改成右（前进）
        #########################
        x_left = 0 if x == 0 else x - 1
        x_right = self.h - 1 if x == self.h - 1 else x + 1
        y_top = self.w - 1 if y == self.w - 1 else y + 1
        y_bottom = 0 if y == 0 else y - 1
        return [(x, y) for x in range(x_left, x_right + 1) for y in range(y_bottom, y_top + 1)]  # 范围3*3领域9个点坐标

    def neight_cost(self, p, next_p):
        # return abs(self.nnum[next_p[0]][next_p[1]] - self.nnum[p[0]][p[1]])
        return self.nnum[next_p[0]][next_p[1]]

    def a_star(self, seed, end):
        process = set()  # 已处理点的集合，集合不能重复
        cost = {seed: 255.0}  # 当前点路径积累的成本代价值
        path = {}  # 路径
        while cost:  # cost为空代表所有点都处理了，每个点处理了其对应的cost值会被删掉
            p = min(cost, key=cost.get)  # 每次取出当前成本代价最小值
            neighbors = self.get_neighbors(p)  # 当前成本代价最小值的领域节点
            process.add(p)  # 保存已处理过的点
            for next_p in [x for x in neighbors if x not in process]:  # 没有被处理过的领域点坐标
                dik_cost = self.neight_cost(p, next_p) + cost[p]  # 当前点与领域的点cost的差值 + 起始点到到当前点累计的cost值
                if next_p in cost:  # 如果该领域点之前计算过了，则需要判断此时所用的代价小还是之前的代价小，如果现在的代价小则需要更新
                    if dik_cost < cost[next_p]:  # 小的话，把之前记录的代价值去除掉。为了之后的更新
                        cost.pop(next_p)
                else:  # 该领域点之前没有计算过 或者 需要更新
                    cost[next_p] = dik_cost  # 该领域所需代价值的更新
                    process.add(next_p)  # 添加到已处理过的点
                    path[next_p] = p  # 把cost最小点作为领域点next_p的前一个点

                if (next_p == end):  # 当前点到达结束点时，提前结束
                    cost = {}  # 为了跳出循环
                    cost[p] = 0  # 为了跳出循环
                    break  # 为了跳出循环
            cost.pop(p)  # 已经处理了的点就排除
        return path

    def item_search(self, cost, item):
        low = 0
        high = len(cost) - 1
        while low <= high:
            middle = (low + high) // 2
            if cost[middle][0] > item:
                high = middle - 1
            elif cost[middle][0] < item:
                low = middle + 1
            else:
                return middle
        return (high + low) // 2 + 1

    def a_star_min_heap(self, seed, end):
        processMap = np.ones(self.nnum.shape)
        # process = set()
        cost = [[255, seed]]
        path = {}
        while cost:
            p = cost[0][1]
            neighbors = self.get_neighbors(p)  # 当前成本代价最小值的领域节点
            processMap[p] = 0
            # process.add(p)  # 保存已处理过的点
            for next_p in [x for x in neighbors if processMap[x] != 0]:  # 没有被处理过的领域点坐标
                dik_cost = self.neight_cost(p, next_p) + cost[0][0]  # 当前点与领域的点cost的差值 + 起始点到到当前点累计的cost值
                # if next_p in cost:  # 如果该领域点之前计算过了，则需要判断此时所用的代价小还是之前的代价小，如果现在的代价小则需要更新
                #     if dik_cost < cost[next_p]:  # 小的话，把之前记录的代价值去除掉。为了之后的更新
                #         cost.pop(next_p)
                # else:  # 该领域点之前没有计算过 或者 需要更新
                cost.insert(self.item_search(cost, dik_cost), [dik_cost, next_p])  # 该领域所需代价值的更新
                # process.add(next_p)  # 添加到已处理过的点
                processMap[next_p] = 0
                path[next_p] = p  # 把cost最小点作为领域点next_p的前一个点

                if (next_p == end):  # 当前点到达结束点时，提前结束
                    cost = []  # 为了跳出循环
                    break  # 为了跳出循环
            if cost:
                cost.pop(0)  # 已经处理了的点就排除
        return path

    def biDijkstra(self, seed, end):
        processSeed = set()
        processEnd = set()
        procseeCross = set()
        costSeed = {seed: 255.0}
        costEnd = {end: 255.0}
        pathSeed = {}
        pathEnd = {}
        pcross = None
        while costSeed:
            pSeed = min(costSeed, key=costSeed.get)
            neighborsPSeed = self.get_neighbors(pSeed)
            processSeed.add(pSeed)
            for next_p in [x for x in neighborsPSeed if x not in processSeed]:
                dik_cost = self.neight_cost(pSeed, next_p) + costSeed[pSeed]
                if next_p not in costSeed:
                    costSeed[next_p] = dik_cost
                    processSeed.add(next_p)
                    pathSeed[next_p] = pSeed
                if next_p in processEnd:
                    procseeCross.add(next_p)
                    for p in [x for x in self.get_neighbors(next_p) if x != next_p]:
                        if p in procseeCross:
                            pcross = next_p
                            costSeed = {}
                            costSeed[pSeed] = 0
                            break
                if (next_p == end):
                    costSeed = {}
                    costSeed[pSeed] = 0
                    break
            costSeed.pop(pSeed)

            pEnd = min(costEnd, key=costEnd.get)
            neighborsPEnd = self.get_neighbors(pEnd)
            processEnd.add(pEnd)
            for next_p in [x for x in neighborsPEnd if x not in processEnd]:
                dik_cost = self.neight_cost(pEnd, next_p) + costEnd[pEnd]
                if next_p not in costEnd:
                    costEnd[next_p] = dik_cost
                    processEnd.add(next_p)
                    pathEnd[next_p] = pEnd
                if next_p in processSeed:
                    procseeCross.add(next_p)
                    for p in [x for x in self.get_neighbors(next_p) if x != next_p]:
                        if p in procseeCross:
                            pcross = next_p
                            costSeed = {}
                            costSeed[pSeed] = 0
                            break
                if (next_p == seed):
                    costSeed = {}
                    costSeed[pSeed] = 0
                    break
            costEnd.pop(pEnd)
        return pcross, pathSeed, pathEnd

    def small_pathPoint(self, seed, end, pcross, pathSeed, pathEnd):
        pathPointSeed = []
        pathPointEnd = []
        pathPointSeed.insert(0, pcross)
        pcrossCopy = copy.copy(pcross)
        pcrossCopy2 = copy.copy(pcross)
        while seed != pcrossCopy:
            top_point = pathSeed[pcrossCopy]  # 更新的top_point为最短路径中某个点的上一个坐标点，即更加靠近种子点
            pathPointSeed.append(top_point)  # 记录路径
            pcrossCopy = top_point

        pathPointSeed = list(reversed(pathPointSeed))

        while end != pcrossCopy2:
            top_point = pathEnd[pcrossCopy2]  # 更新的top_point为最短路径中某个点的上一个坐标点，即更加靠近种子点
            pathPointEnd.append(top_point)  # 记录路径
            pcrossCopy2 = top_point

        pathPointSeed.extend(pathPointEnd)
        return pathPointSeed

    def small_path_point(self, seed, end, paths):
        path_piont = []
        path_piont.insert(0, end)  # 把结束点加到路径中
        while seed != end:  # 直到结束点坐标等于开始点坐标是结束
            top_point = paths[end]  # 更新的top_point为最短路径中某个点的上一个坐标点，即更加靠近种子点
            path_piont.append(top_point)  # 记录路径
            end = top_point  # 更新点坐标
        return path_piont


if __name__ == '__main__':
    pass

    # start = time.time()
    # dijkstra = Dijkstra()
    # nnum = shuzu(300, 400)  # 创建二维数组
    # seed = (0, 0)  # 起始点
    # end = (200, 200)  # 结束点
    # h = nnum.shape[1]  # 高
    # w = nnum.shape[0]  # 宽
    # print('地图\n', nnum)  # 显示地图
    # print('起始点', seed)
    #
    # paths = a_star(nnum, seed, end)
    # print('dijkstra所有路径', paths)
    # path_piont = small_path_point(seed, end, paths)  # 开始点到结束点的最短路径所经过的坐标点
    # print('起始点:', seed, '到结束点:', end, '的dijkstra最短路径为:', path_piont)
    #
    # print('一共走了%d步' % len(path_piont))
    # all_leng = 0
    # for i in range(len(path_piont) - 1):
    #     leng = nnum[path_piont[i]] + nnum[path_piont[i + 1]]
    #     all_leng = leng + all_leng
    # print('权重:', all_leng)
    #
    # end = time.time()
    # print('总共耗时:', end - start)
