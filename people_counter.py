import cv2
import numpy as np

class PeopleCounter:
    def __init__(self, debug_mode=False):
        # カウントの初期化
        self.entry_count = 0
        self.exit_count = 0
        self.current_inside = 0
        # トラッカーの初期化
        self.trackers = cv2.legacy.MultiTracker_create()
        # エリアラインの設定
        self.line_position = 250  # 調整が必要
        self.direction_threshold = 5
        # デバッグモードフラグ
        self.debug_mode = debug_mode
        # オブジェクトID管理
        self.object_id = 0
        self.objects = {}
        # YOLOの初期化
        self.net, self.output_layers, self.classes = self.initialize_yolo()

    def initialize_yolo(self):
        # モデルとクラス名のパス
        model_cfg = 'yolo/yolov4.cfg'
        model_weights = 'yolo/yolov4.weights'
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
                if self.classes[class_id] == 'person' and confidence > 0.5:
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detections = []

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            detections.append((x, y, w, h))

            if self.debug_mode:
                # バウンディングボックスの描画
                label = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections, frame

    def process_frame(self, frame):
        # YOLOで人物検出
        detections, frame = self.detect_people(frame)

        # トラッカーの更新
        success, boxes = self.trackers.update(frame)

        new_trackers = cv2.legacy.MultiTracker_create()
        new_objects = {}

        for i, box in enumerate(boxes):
            x, y, w, h = [int(v) for v in box]
            center_y = y + h // 2
            object_id = list(self.objects.keys())[i]
            prev_center_y = self.objects[object_id]

            # 入退室の判定
            if prev_center_y < self.line_position and center_y >= self.line_position:
                self.entry_count += 1
                self.current_inside += 1
            elif prev_center_y > self.line_position and center_y <= self.line_position:
                self.exit_count += 1
                self.current_inside = max(0, self.current_inside - 1)

            # オブジェクトの中心位置を更新
            new_objects[object_id] = center_y

            # トラッカーの追加
            tracker = cv2.legacy.TrackerKCF_create()
            new_trackers.add(tracker, frame, box)

            if self.debug_mode:
                # バウンディングボックスとIDの表示
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                # 中心点の描画
                cv2.circle(frame, (x + w // 2, center_y), 5, (0, 255, 255), -1)

        # 新しい検出をトラッカーに追加
        for detection in detections:
            x, y, w, h = detection
            self.object_id += 1
            center_y = y + h // 2
            new_objects[self.object_id] = center_y
            tracker = cv2.legacy.TrackerKCF_create()
            tracker.init(frame, detection)
            new_trackers.add(tracker, frame, detection)

        self.trackers = new_trackers
        self.objects = new_objects

        if self.debug_mode:
            # エリアラインの描画
            cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 255, 255), 2)

        return frame
