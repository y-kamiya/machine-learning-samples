from __future__ import print_function
import sys
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from google.cloud import translate_v3beta1 as translate


class SpreadSheet:
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

    # The ID and range of a sample spreadsheet.
    SAMPLE_SPREADSHEET_ID = '1YC03OZZ5e3F8MWz6gyAzxfZevyOJy0lBDo_Smn94gHk'
    SAMPLE_RANGE_NAME = ['H:H', 'V:V', 'W:W']

    def __init__(self):
        creds = self._get_creds()
        assert creds is not None, 'can not get correct credentials'

        service = build('sheets', 'v4', credentials=creds)

        self.sheet = service.spreadsheets()
        self.rowIndexes = []
        self.contents = []

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
        result = self.sheet.values().batchGet(spreadsheetId=self.SAMPLE_SPREADSHEET_ID,
                                    ranges=self.SAMPLE_RANGE_NAME).execute()

        print(result)
        valueRanges = result.get('valueRanges', [])
        sources = valueRanges[0].get('values', [])
        ids = valueRanges[1].get('values', [])
        targets = valueRanges[2].get('values', [])

        for i in range(len(sources)):
            if len(sources[i]) == 0:
                continue
            if i < len(targets) and len(targets[i]) != 0:
                continue

            self.rowIndexes.append(i+1)
            self.contents.append(sources[i][0])

        print(self.rowIndexes, self.contents)

        return self.contents

    def write(self, texts):
        assert len(self.rowIndexes) == len(texts), "translated texts should be same length with row indexes"

        data = []
        for i in range(len(texts)):
            print('row: {}, {}'.format(self.rowIndexes[i], texts[i]))

            rangeValue = 'W{}'.format(self.rowIndexes[i])
            data.append({'range': rangeValue, 'values': [[texts[i]]]})

        body = {
            'valueInputOption': 'RAW',
            'data': data
        }
        print(body)
        response = self.sheet.values().batchUpdate(spreadsheetId=self.SAMPLE_SPREADSHEET_ID, body=body).execute()
        print(response)

    def copy_sheet(self):
        original_sheet_id = 0

        body = {
            'destination_spreadsheet_id': self.SAMPLE_SPREADSHEET_ID,
        }

        request = self.sheet.sheets().copyTo(spreadsheetId=self.SAMPLE_SPREADSHEET_ID, sheetId=original_sheet_id, body=body)
        response = request.execute()

        print(response)

class Translation:
    def __init__(self, src_texts):
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
            source_language_code='ja-JP',
            target_language_code='en-US')

        results = [entry.translated_text for entry in response.translations]
        # results = [entry.translated_text.replace('\n', '\\n') for entry in response.translations]
        # results = ["I can't see data even with my identification skills\\nIn strength, it is probably the 90th class ...", "Because I earn time, get away quickly!"]
        print(results)

        return results


if __name__ == '__main__':
    sheet = SpreadSheet()
    #sheet.copy_sheet()

    src_texts = sheet.get_src_texts()

    if len(src_texts) == 0:
        print('no translation text is found')
        sys.exit()

    translation = Translation(src_texts)
    tgt_texts = translation.get_tgt_texts()

    sheet.write(tgt_texts)

