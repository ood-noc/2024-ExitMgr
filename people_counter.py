import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class PeopleCounter:
    def __init__(self, debug_mode=False):
        # カウントの初期化
        self.entry_count = 0
        self.exit_count = 0
        self.current_inside = 0
        # トラッカーの初期化（辞書に変更）
        self.trackers = {}
        # エリアラインの設定（縦線）
        self.line_position = 320  # フレーム幅に合わせて調整
        # デバッグモードフラグ
        self.debug_mode = debug_mode
        # オブジェクトID管理
        self.object_id = 0
        self.objects = {}
        # トラッカーの生存期間管理
        self.tracker_lifetimes = {}
        # YOLOの初期化（Tiny-YOLOv4を使用）
        self.net, self.output_layers, self.classes = self.initialize_yolo()

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
                if len(detection) < 7:
                    continue  # 異常なデータをスキップ
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
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # 検出がない場合の処理
            pass

        return detections, frame

    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IOU) between two bounding boxes.
        Each box is a tuple (x, y, w, h)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate coordinates of intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = w1 * h1
        box2_area = w2 * h2

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    def calculate_distance(self, center1, center2):
        """
        Calculate Euclidean distance between two centers.
        Each center is a tuple (x, y)
        """
        return np.linalg.norm(np.array(center1) - np.array(center2))

    def process_frame(self, frame):
        # YOLOで人物検出
        detections, frame = self.detect_people(frame)

        # トラッカーの更新
        updated_trackers = {}
        updated_objects = {}
        tracker_boxes = {}
        tracker_centers = {}

        # 既存トラッカーの更新と情報取得
        for object_id, tracker in self.trackers.items():
            ok, box = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in box]
                center = (x + w // 2, y + h // 2)
                tracker_boxes[object_id] = (x, y, w, h)
                tracker_centers[object_id] = center
                updated_objects[object_id] = center[0]  # X coordinate
                self.tracker_lifetimes[object_id] = 0  # reset lifetime
            else:
                # Increment lifetime
                self.tracker_lifetimes[object_id] = self.tracker_lifetimes.get(object_id, 0) + 1
                if self.tracker_lifetimes[object_id] > 5:
                    if self.debug_mode:
                        print(f"Tracker {object_id} removed after exceeding misses.")
                    continue  # do not add to updated_trackers

        # マッチングの準備
        unmatched_detections = detections.copy()
        matched_trackers = set()
        matches = []

        # トラッカーIDのリスト
        tracker_ids = list(tracker_boxes.keys())

        # IOUマトリックスの計算
        iou_matrix = []
        detection_boxes = []
        for det in detections:
            det_box = det
            detection_boxes.append(det_box)
            ious = []
            for trk_id in tracker_ids:
                trk_box = tracker_boxes[trk_id]
                iou = self.compute_iou(det_box, trk_box)
                ious.append(iou)
            iou_matrix.append(ious)

        if len(iou_matrix) > 0 and len(tracker_ids) > 0:
            iou_matrix = np.array(iou_matrix)
            # Hungarianアルゴリズムのため、コスト行列として1 - IOUを使用
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_ind, col_ind):
                if iou_matrix[row][col] >= 0.3:
                    det = detection_boxes[row]
                    trk_id = tracker_ids[col]
                    matches.append((det, trk_id))
                    matched_trackers.add(trk_id)
                    if det in unmatched_detections:
                        unmatched_detections.remove(det)

        # マッチングされたトラッカーの更新
        for det, trk_id in matches:
            x, y, w, h = det
            center_x = x + w // 2
            updated_objects[trk_id] = center_x
            updated_trackers[trk_id] = self.trackers[trk_id]

        # マッチングされなかった検出に対する新規トラッカーの作成
        for det in unmatched_detections:
            x, y, w, h = det
            center_det = (x + w // 2, y + h // 2)
            # 既存トラッカーとの距離をチェック
            close = False
            for trk_id, trk_center in tracker_centers.items():
                distance = self.calculate_distance(center_det, trk_center)
                if distance < 50:  # 距離の閾値を調整
                    close = True
                    break
            if not close:
                self.object_id += 1
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(frame, tuple(det))
                updated_trackers[self.object_id] = tracker
                updated_objects[self.object_id] = center_det[0]
                self.tracker_lifetimes[self.object_id] = 0

        # マッチングされなかった既存トラッカーの保持
        for object_id in self.trackers.keys():
            if object_id not in matched_trackers and object_id in tracker_boxes:
                updated_trackers[object_id] = self.trackers[object_id]
                updated_objects[object_id] = tracker_centers[object_id][0]

        # カウントの更新
        for object_id in updated_trackers.keys():
            if object_id in self.objects:
                prev_center_x = self.objects[object_id]
                curr_center_x = updated_objects[object_id]

                # 入退室の判定（X軸方向）
                if prev_center_x < self.line_position and curr_center_x >= self.line_position:
                    self.entry_count += 1
                    self.current_inside += 1
                elif prev_center_x > self.line_position and curr_center_x <= self.line_position:
                    self.exit_count += 1
                    self.current_inside = max(0, self.current_inside - 1)

        # トラッカーとオブジェクト情報を更新
        self.trackers = updated_trackers
        self.objects = updated_objects

        if self.debug_mode:
            # デバッグ情報の表示
            for object_id, tracker in self.trackers.items():
                if object_id in tracker_boxes:
                    x, y, w, h = tracker_boxes[object_id]
                    # バウンディングボックスとIDの表示（青色）
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f'ID: {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    # 中心点の描画
                    center = tracker_centers[object_id]
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)

            # エリアラインの描画（垂直線）
            cv2.line(frame, (self.line_position, 0), (self.line_position, frame.shape[0]), (0, 255, 255), 2)

            # カウント情報の表示
            cv2.putText(frame, f"Entry: {self.entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Exit: {self.exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Inside: {self.current_inside}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame
