# centroid_tracker.py

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_distance=50, max_frames=5):
        # 次に割り当てるオブジェクトID
        self.next_object_id = 0
        # 現在のオブジェクトIDとそのセントロイドのマッピング
        self.objects = OrderedDict()
        # オブジェクトが失踪したフレーム数のカウンター
        self.disappeared = OrderedDict()

        # パラメータ
        self.max_distance = max_distance
        self.max_frames = max_frames

    def register(self, centroid):
        """
        新しいオブジェクトを登録する
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        オブジェクトを登録解除する
        """
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        """
        セントロイドのリストを受け取り、オブジェクトのトラッキングを更新する
        """
        # 入力されたセントロイドがない場合
        if len(input_centroids) == 0:
            # 既存のオブジェクトすべての失踪カウンターを増加
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                # 失踪カウンターが最大値を超えたらオブジェクトを登録解除
                if self.disappeared[object_id] > self.max_frames:
                    self.deregister(object_id)

            return self.objects

        # 現在のオブジェクトがない場合、すべて新規登録
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # 現在のオブジェクトIDとセントロイドのリストを取得
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # セントロイド間の距離行列を計算
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # 最小距離のインデックスを取得
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # 既にマッチングされた行と列のセット
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                # 既にマッチングされている場合はスキップ
                if row in used_rows or col in used_cols:
                    continue

                # セントロイド間の距離が閾値を超える場合はスキップ
                if D[row, col] > self.max_distance:
                    continue

                # オブジェクトを更新
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # 使用済みの行と列を記録
                used_rows.add(row)
                used_cols.add(col)

            # マッチングされなかった行と列を取得
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # オブジェクト数 >= セントロイド数の場合
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    # 失踪カウンターが最大値を超えたら登録解除
                    if self.disappeared[object_id] > self.max_frames:
                        self.deregister(object_id)
            else:
                # 新しいオブジェクトを登録
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects
