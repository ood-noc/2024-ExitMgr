import gspread
from google.oauth2.service_account import Credentials

class SpreadsheetManager:
    def __init__(self, service_account_file, spreadsheet_url, sheet_name=None):
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

    def append_row(self, row_data):
        # データを1行追加
        self.worksheet.append_row(row_data)