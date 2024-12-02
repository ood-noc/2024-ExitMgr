import cv2
import numpy as np
from collections import deque

class PeopleCounter:
    def __init__(self, debug_mode=False, max_distance=50, max_frames=5):
        # カウントの初期化
        self.entry_count = 0
        self.exit_count = 0
        self.current_inside = 0

        # エリアラインの設定（縦線）
        self.line_position = 320  # フレーム幅に合わせて調整

        # デバッグモードフラグ
        self.debug_mode = debug_mode

        # YOLOの初期化（Tiny-YOLOv4を使用）
        self.net, self.output_layers, self.classes = self.initialize_yolo()

        # オブジェクトの追跡用データ構造
        self.tracked_objects = {}
        self.next_object_id = 0

        # セントロイドの移動履歴を保持
        self.max_distance = max_distance  # セントロイド間の最大距離
        self.max_frames = max_frames      # 同一人物と判断するための最大フレーム数

    def initialize_yolo(self):
        # モデルとクラス名のパス（Tinyモデルを使用）
        model_cfg = 'yolo/yolov4-tiny.cfg'
        model_weights = 'yolo/yolov4-tiny.weights'
        classes_file = 'yolo/coco.names'

        # クラス名の読み込み
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # ネットワークの読み込み
        net = cv2.dnn.readNet(model_weights, model_cfg)

        # CPUを使用する場合（GPUを使用する場合は設定変更が必要）
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 出力レイヤーの取得
        layer_names = net.getUnconnectedOutLayersNames()

        return net, layer_names, classes

    def detect_people(self, frame):
        height, width, _ = frame.shape

        # 入力画像の作成
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        # 推論の実行
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # 推論結果の解析
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # クラスIDが「person」であり、信頼度がしきい値を超える場合
                if self.classes[class_id] == 'person' and confidence > 0.3:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 重複するボックスの除去
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

        detections = []

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                detections.append((x, y, w, h))

                if self.debug_mode:
                    # バウンディングボックスの描画（緑色）
                    label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 検出がない場合の処理
            pass

        return detections, frame

    def compute_centroid(self, box):
        x, y, w, h = box
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        return (center_x, center_y)

    def process_frame(self, frame):
        # YOLOで人物検出
        detections, frame = self.detect_people(frame)

        current_centroids = [self.compute_centroid(det) for det in detections]
        updated_tracked_objects = {}

        # 更新されたオブジェクトの移動履歴を保持
        for centroid in current_centroids:
            matched = False
            for object_id, data in self.tracked_objects.items():
                last_centroid = data['centroids'][-1]
                distance = np.linalg.norm(np.array(centroid) - np.array(last_centroid))
                if distance < self.max_distance:
                    # 同一人物と判断
                    updated_tracked_objects[object_id] = {
                        'centroids': data['centroids'] + [centroid],
                        'frames_since_update': 0
                    }
                    matched = True

                    # カウントの更新
                    if len(updated_tracked_objects[object_id]['centroids']) >= 2:
                        prev_x = updated_tracked_objects[object_id]['centroids'][-2][0]
                        curr_x = updated_tracked_objects[object_id]['centroids'][-1][0]

                        if prev_x < self.line_position and curr_x >= self.line_position:
                            self.entry_count += 1
                            self.current_inside += 1
                        elif prev_x > self.line_position and curr_x <= self.line_position:
                            self.exit_count += 1
                            self.current_inside = max(0, self.current_inside - 1)

                    break

            if not matched:
                # 新しいオブジェクトとして登録
                updated_tracked_objects[self.next_object_id] = {
                    'centroids': [centroid],
                    'frames_since_update': 0
                }
                self.next_object_id += 1

        # オブジェクトが検出されなかった場合の処理
        for object_id, data in self.tracked_objects.items():
            if object_id not in updated_tracked_objects:
                data['frames_since_update'] += 1
                if data['frames_since_update'] < self.max_frames:
                    updated_tracked_objects[object_id] = data
                else:
                    # 一定フレーム数更新されない場合は削除
                    pass

        self.tracked_objects = updated_tracked_objects

        if self.debug_mode:
            # デバッグ情報の表示
            for object_id, data in self.tracked_objects.items():
                centroid = data['centroids'][-1]
                # セントロイドの描画
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
                cv2.putText(frame, f'ID: {object_id}', (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # エリアラインの描画（縦線）
            cv2.line(frame, (self.line_position, 0), (self.line_position, frame.shape[0]), (255, 0, 255), 2)

            # カウント情報の表示
            cv2.putText(frame, f"Entry: {self.entry_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Exit: {self.exit_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Inside: {self.current_inside}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame
