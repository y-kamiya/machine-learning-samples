from __future__ import print_function
import sys
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from google.cloud import translate_v3beta1 as translate

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '1YC03OZZ5e3F8MWz6gyAzxfZevyOJy0lBDo_Smn94gHk'
SAMPLE_RANGE_NAME = ['H:H', 'V:V', 'W:W']

def main():
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
                'keys/client_secret_other.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().batchGet(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                ranges=SAMPLE_RANGE_NAME).execute()

    print(result)
    valueRanges = result.get('valueRanges', [])
    sources = valueRanges[0].get('values', [])
    ids = valueRanges[1].get('values', [])
    targets = valueRanges[2].get('values', [])




    rowIndexes = []
    contents = []
    for i in range(len(sources)):
        if len(sources[i]) == 0:
            continue
        if i < len(targets) and len(targets[i]) != 0:
            continue

        rowIndexes.append(i+1)
        contents.append(sources[i][0])

    print(rowIndexes, contents)
    if len(contents) == 0:
        print('no translation text is found')
        sys.exit()

    # translation api
    client = translate.TranslationServiceClient()
    project_id = 'fluid-axe-251004'
    location = 'global'
    parent = client.location_path(project_id, location)

    response = client.translate_text(
        parent=parent,
        contents=contents,
        mime_type='text/plain',  # mime types: text/plain, text/html
        source_language_code='ja-JP',
        target_language_code='en-US')

    results = [entry.translated_text for entry in response.translations]
    # results = [entry.translated_text.replace('\n', '\\n') for entry in response.translations]
    # results = ["I can't see data even with my identification skills\\nIn strength, it is probably the 90th class ...", "Because I earn time, get away quickly!"]
    print(results)

    data = []
    for i in range(len(results)):
        print('row: {}, {}'.format(rowIndexes[i], results[i]))

        rangeValue = 'W{}'.format(rowIndexes[i])
        data.append({'range': rangeValue, 'values': [[results[i]]]})


    body = {
        'valueInputOption': 'RAW',
        'data': data
    }
    print(body)
    response = sheet.values().batchUpdate(spreadsheetId=SAMPLE_SPREADSHEET_ID, body=body).execute()
    print(response)





if __name__ == '__main__':
    main()
