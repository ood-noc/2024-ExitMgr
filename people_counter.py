import cv2
import numpy as np

class PeopleCounter:
    def __init__(self, debug_mode=False):
        # 背景差分器の初期化
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        # カウントの初期化
        self.entry_count = 0
        self.exit_count = 0
        self.current_inside = 0
        # トラッカーの初期化（legacy モジュールを使用）
        self.trackers = cv2.legacy.MultiTracker_create()
        # エリアラインの設定
        self.line_position = 250
        self.direction_threshold = 5
        # デバッグモードフラグ
        self.debug_mode = debug_mode

    def process_frame(self, frame):
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 背景差分
        fg_mask = self.bg_subtractor.apply(gray)

        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # 輪郭検出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:
            # 小さな輪郭は無視
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

        # トラッカーの更新
        success, boxes = self.trackers.update(frame)

        # 新しいトラッカーの初期化（legacy モジュールを使用）
        new_trackers = cv2.legacy.MultiTracker_create()
        for i, box in enumerate(boxes):
            x, y, w, h = [int(v) for v in box]
            center_y = y + h // 2

            # 入退室の判定
            if center_y < self.line_position - self.direction_threshold:
                self.exit_count += 1
                self.current_inside = max(0, self.current_inside - 1)
            elif center_y > self.line_position + self.direction_threshold:
                self.entry_count += 1
                self.current_inside += 1
            else:
                # トラッカーの追加（legacy モジュールを使用）
                new_trackers.add(cv2.legacy.TrackerKCF_create(), frame, box)
                if self.debug_mode:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 新しい検出をトラッカーに追加
        for detection in detections:
            # トラッカーの作成（legacy モジュールを使用）
            tracker = cv2.legacy.TrackerKCF_create()
            self.trackers.add(tracker, frame, detection)

        # トラッカーを更新
        self.trackers = new_trackers

        if self.debug_mode:
            # エリアラインの描画
            cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 255, 255), 2)

        return frame
