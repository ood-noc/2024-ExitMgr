import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials

class SpreadsheetManager:
    def __init__(self, service_account_file, spreadsheet_url, sheet_name=None):
        self.is_authenticated = False
        try: 
            # サービスアカウントの認証
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            credentials = Credentials.from_service_account_file(service_account_file, scopes=scopes)
            self.client = gspread.authorize(credentials)
            # スプレッドシートにアクセス
            self.spreadsheet = self.client.open_by_url(spreadsheet_url)
            # ワークシートを指定（デフォルトは最初のシート）
            if sheet_name:
                self.worksheet = self.spreadsheet.worksheet(sheet_name)
            else:
                self.worksheet = self.spreadsheet.sheet1
            
            self.is_authenticated = True
            print("スプレッドシートへの認証とアクセスに成功しました")

        except (FileNotFoundError, gspread.exceptions.APIError, Exception) as e:
            print(f"認証またはスプレッドシートへのアクセスに失敗しました: {e}")
            self.is_authenticated = False

    def append_row_with_timestamp(self, row_data):
        if not self.is_authenticated:
            print("認証に失敗しているため、データの書き込みをスキップします")
            return
        try:
            # 現在のタイムスタンプを取得
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # タイムスタンプをデータの先頭に追加
            row_with_timestamp = [timestamp] + row_data
            # データを1行追加
            self.worksheet.append_row(row_with_timestamp)
        except gspread.exceptions.APIError as e:
            print(f"データの書き込みに失敗しました: {row_with_timestamp} - エラー: {e}")

    def append_row(self, row_data):
        # データを1行追加
        self.worksheet.append_row(row_data)