import cv2
from people_counter import PeopleCounter
import time
import argparse

def main():
    # コマンドライン引数の処理
    parser = argparse.ArgumentParser(description="People Counting System")
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグモード（映像を表示）')
    args = parser.parse_args()

    # カメラの設定（デバイスIDを適宜変更）
    cap = cv2.VideoCapture(0)

    # PeopleCounterの初期化
    pc = PeopleCounter(debug_mode=args.debug)

    # 時間計測のための初期化
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # フレームのリサイズ（処理速度向上のため）
        frame = cv2.resize(frame, (640, 480))

        # 人物検出とカウント
        frame = pc.process_frame(frame)

        # カウントの取得
        entry_count = pc.entry_count
        exit_count = pc.exit_count
        current_inside = pc.current_inside

        # 10秒ごとに標準出力に表示
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:
            print(f"Entry: {entry_count}, Exit: {exit_count}, Inside: {current_inside}")
            start_time = time.time()  # タイマーをリセット

        if args.debug:
            # カウント情報をフレームに表示
            cv2.putText(frame, f"Entry: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Exit: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Inside: {current_inside}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
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
