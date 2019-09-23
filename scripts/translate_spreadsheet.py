from __future__ import print_function
import sys
import pickle
import os.path
import argparse
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from google.cloud import translate_v3beta1 as translate


class SpreadSheet:
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    SRC_COLUMNS = {
        'ja': 'H',
        'en': 'M',
        'fr': 'N',
        'de': 'O',
        'es': 'P',
        'tc': 'Q',
        'kr': 'R',
    }
    TGT_COLUMNS = {
        'ja': {
            'en': 'V',
            'fr': 'W',
            'de': 'Y',
            'es': 'AA',
            'tc': 'AC',
            'kr': 'AE',
        },
        'en': {
            'fr': 'X',
            'de': 'Z',
            'es': 'AB',
            'tc': 'AD',
        },
    }

    # SPREADSHEET_ID = '1YC03OZZ5e3F8MWz6gyAzxfZevyOJy0lBDo_Smn94gHk'
    SPREADSHEET_ID = '1e2oj9ciwCxpSKp3IupETJuukjrOmlh6MoDzkpZ2C_ow'
    SHEET_NAME = 'マスタ翻訳依頼'

    def __init__(self, config):
        src = config.src
        tgt = config.tgt
        assert src in self.SRC_COLUMNS, 'src language is not supported'
        assert src in self.TGT_COLUMNS, 'src language is not supported'
        assert tgt in self.TGT_COLUMNS[src], 'tgt language is not supported'

        creds = self._get_creds()
        assert creds is not None, 'can not get correct credentials'

        print('spreadsheet id: {}, sheet name: {}'.format(self.SPREADSHEET_ID, self.SHEET_NAME))

        service = build('sheets', 'v4', credentials=creds)

        self.config = config
        self.sheet = service.spreadsheets()
        self.rowIndexes = []
        self.contents = []


    def _build_range_name_to_get(self):
        src_column = self.SRC_COLUMNS[self.config.src]
        tgt_column = self.TGT_COLUMNS[self.config.src][self.config.tgt]

        range_name = [
            '{}!{}{}:{}'.format(self.SHEET_NAME, src_column, self.config.start_row, src_column),
            '{}!{}{}:{}'.format(self.SHEET_NAME, tgt_column, self.config.start_row, tgt_column),
        ]
        print('range: {}'.format(range_name))

        return range_name

    def _get_creds(self):
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'keys/client_secret_other.json', self.SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return creds

    def get_src_texts(self):
        # Call the Sheets API
        result = self.sheet.values().batchGet(spreadsheetId=self.SPREADSHEET_ID,
                                    ranges=self._build_range_name_to_get()).execute()

        # print(result)
        valueRanges = result.get('valueRanges', [])
        sources = valueRanges[0].get('values', [])
        targets = valueRanges[1].get('values', [])

        for i in range(len(sources)):
            if len(sources[i]) == 0:
                continue
            if i < len(targets) and len(targets[i]) != 0:
                continue

            self.rowIndexes.append(i + self.config.start_row)
            self.contents.append(sources[i][0])

        # print(self.rowIndexes, self.contents)

        return self.contents

    def write(self, texts):
        assert len(self.rowIndexes) == len(texts), "translated texts should be same length with row indexes"

        src = self.config.src
        tgt = self.config.tgt

        data = []
        for i in range(len(texts)):
            print('row: {}, {}'.format(self.rowIndexes[i], texts[i]))

            rangeValue = '{}!{}{}'.format(self.SHEET_NAME, self.TGT_COLUMNS[src][tgt], self.rowIndexes[i])
            data.append({'range': rangeValue, 'values': [[texts[i]]]})

        body = {
            'valueInputOption': 'RAW',
            'data': data
        }
        print(body)
        response = self.sheet.values().batchUpdate(spreadsheetId=self.SPREADSHEET_ID, body=body).execute()
        print(response)

    def copy_sheet(self):
        original_sheet_id = 0

        body = {
            'destination_spreadsheet_id': self.SPREADSHEET_ID,
        }

        request = self.sheet.sheets().copyTo(spreadsheetId=self.SPREADSHEET_ID, sheetId=original_sheet_id, body=body)
        response = request.execute()

        print(response)

class Translation:
    LANG_CODE_MAP = {
        'ja': 'ja',
        'en': 'en',
        'fr': 'fr',
        'de': 'de',
        'es': 'es',
        'tc': 'zh-TW',
        'kr': 'ko',
    }

    def __init__(self, config, src_texts):
        assert config.src in self.LANG_CODE_MAP, 'src language is not supported'
        assert config.tgt in self.LANG_CODE_MAP, 'tgt language is not supported'

        self.config = config
        self.src_texts = src_texts

    def get_tgt_texts(self):
        # translation api
        client = translate.TranslationServiceClient()
        project_id = 'fluid-axe-251004'
        location = 'global'
        parent = client.location_path(project_id, location)

        response = client.translate_text(
            parent=parent,
            contents=self.src_texts,
            mime_type='text/plain',  # mime types: text/plain, text/html
            source_language_code=self.LANG_CODE_MAP[self.config.src],
            target_language_code=self.LANG_CODE_MAP[self.config.tgt])

        results = [entry.translated_text for entry in response.translations]
        # results = [entry.translated_text.replace('\n', '\\n') for entry in response.translations]
        # results = ["I can't see data even with my identification skills\\nIn strength, it is probably the 90th class ...", "Because I earn time, get away quickly!"]
        # print(results)

        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--src', default='ja', help='src language code')
    parser.add_argument('--tgt', default='en', help='tgt language code')
    parser.add_argument('--start_row', type=int, default=1, help='translate after this row index')
    parser.add_argument('--backup', action='store_true', help='copy original sheet before processed')
    args = parser.parse_args()

    print('------------------------- start translation process -------------------------')
    print(args)
    
    sheet = SpreadSheet(args)
    if (args.backup):
        sheet.copy_sheet()

    src_texts = sheet.get_src_texts()

    if len(src_texts) == 0:
        print('no translation text is found')
        sys.exit()

    translation = Translation(args, src_texts)
    tgt_texts = translation.get_tgt_texts()

    sheet.write(tgt_texts)
    print('------------------------- end translation process -------------------------')

