# main.py

import cv2
from people_counter import PeopleCounter
from spreadsheet import SpreadsheetManager
import time
import argparse
import yaml

def main():
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description="People Counting System with Centroid Tracking")
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード（映像を表示）')
    args = parser.parse_args()

    # YAMLファイルから設定値を読み込む
    with open('secrets/settings.yml', 'r') as file:
        config = yaml.safe_load(file)

    # 設定値を取得
    service_account_file = config['service_account_file']
    spreadsheet_url = config['spreadsheet_url']
    sheet_name = config.get('sheet_name')  # sheet_nameはオプション

    # カメラの設定（デバイスIDを適宜変更）
    cap = cv2.VideoCapture(0)

    # フレームサイズの設定（処理速度向上のため）
    frame_width = 640
    frame_height = 480

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # PeopleCounterの初期化
    pc = PeopleCounter(debug_mode=args.debug)

    # SpreadsheetManagerの初期化
    manager = SpreadsheetManager(service_account_file, spreadsheet_url, sheet_name)

    # 時間計測のための初期化
    start_time = time.time()

    # FPSカウンターの初期化
    frame_count = 0
    fps = 0
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # フレームのリサイズ（必要に応じて調整）
        frame = cv2.resize(frame, (frame_width, frame_height))

        # 人物検出とカウント
        frame = pc.process_frame(frame)

        # フレームカウントとFPS計算
        frame_count += 1
        elapsed_fps_time = time.time() - fps_time
        if elapsed_fps_time >= 1.0:  # 1秒ごとにFPSを計算
            fps = frame_count / elapsed_fps_time
            frame_count = 0
            fps_time = time.time()

        # FPSの表示（右上に配置）
        if args.debug:
            text = f"FPS: {fps:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = frame_width - text_size[0] - 10  # 右端から10px内側
            text_y = 30  # 上端から30px下
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # カウントの取得
        entry_count = pc.entry_count
        exit_count = pc.exit_count
        current_inside = pc.current_inside
        
        # TODO: 後で書き換える
        device_id = 1

        # 10秒ごとに標準出力に表示
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:
            print(f"Entry: {entry_count}, Exit: {exit_count}, Inside: {current_inside}, FPS: {fps:.2f}")
            # スプレッドシートに記録
            manager.append_row_with_timestamp([device_id, entry_count, exit_count, current_inside])
            start_time = time.time()  # タイマーをリセット

        if args.debug:
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            # 非デバッグモードでもキー入力を処理
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args.debug:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
