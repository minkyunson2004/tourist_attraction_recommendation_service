# /app.py
from flask import Flask, render_template, request
import torch
from torch import nn
import pandas as pd
import json
import requests


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }
        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003', headers=headers, json=completion_request, stream=True) as r:
            i = 0
            description = ''
            for line in r.iter_lines():
                if i % 4 == 2:
                    data = line.decode('utf-8')
                    data = data[5:]
                    json_data = json.loads(data)
                    if data[2] != 'd' and json_data['outputLength'] != 1:
                        description = json_data['message']['content']
                        print(description)
                i += 1
        return description


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(1032, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1)
        )

    def forward(self, x):
        return self.sequential(x)


completion_executor = CompletionExecutor(
    host='https://clovastudio.stream.ntruss.com',
    api_key='api_key',
    api_key_primary_val='api_key_primary_val',
    request_id='request_id'
)


def make_description(area_name):
    preset_text = [
        {
            "role": "system",
            "content": "- 사용자가 입력한 관광지에 대한 설명을 출력합니다."
        },
        {
            "role": "user",
            "content": f"{area_name}"
        }
    ]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 512,
        'temperature': 0.5,
        'repeatPenalty': 1.2,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    return completion_executor.execute(request_data)


# 모델 불러오기
model = RegressionModel()
model.load_state_dict(torch.load('model_training/model.pth'))
model.eval()

# Flask 객체 인스턴스 생성
app = Flask(__name__)


@app.route('/', methods=('GET', 'POST'))
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def results():
    gender = request.form['gender']
    age = request.form['age']
    travel_accompany = request.form['travel_accompany']
    travel_style1 = request.form['travel_style1']
    travel_style2 = request.form['travel_style2']
    travel_style3 = request.form['travel_style3']
    travel_style4 = request.form['travel_style4']

    gender = int(gender)
    age = int(age)
    travel_accompany = int(travel_accompany)
    travel_style1 = int(travel_style1)
    travel_style2 = int(travel_style2)
    travel_style3 = int(travel_style3)
    travel_style4 = int(travel_style4)

    df = pd.DataFrame(columns=['Name', 'DGSTFN'])
    place = pd.read_csv('data_preprocessing/place_list.csv')
    embedding = pd.read_csv('data_preprocessing/place_name_to_embedding.csv')

    for idx in place.index:
        x = [gender, age, travel_accompany, travel_style1, travel_style2, travel_style3, travel_style4, place['VISIT_AREA_TYPE_CD'][idx]]
        x = x + list(embedding[place['VISIT_AREA_NM'][idx]])
        x = torch.FloatTensor(x)
        x.reshape(1, x.shape[0])
        with torch.no_grad():
            prediction = model(x)
            if idx == 0:
                df = pd.DataFrame({'Name': place['VISIT_AREA_NM'][idx], 'DGSTFN': prediction})
            else:
                new_df = pd.DataFrame({'Name': place['VISIT_AREA_NM'][idx], 'DGSTFN': prediction})
                df = pd.concat([df, new_df], ignore_index=True)

    df.sort_values(by=['DGSTFN'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    place_list = df['Name'][:100]
    place_list = place_list.sample(frac=1)
    place_list = list(place_list)[:5]

    place_description = []
    for place in place_list:
        place_description.append(make_description(place))

    return render_template('result.html',
                           place0=place_list[0],
                           place1=place_list[1],
                           place2=place_list[2],
                           place3=place_list[3],
                           place4=place_list[4],
                           place_description0=place_description[0],
                           place_description1=place_description[1],
                           place_description2=place_description[2],
                           place_description3=place_description[3],
                           place_description4=place_description[4])


if __name__ == "__main__":
    app.run(host="127.0.0.1")
    # host 등을 직접 지정하고 싶다면
    # app.run(host="127.0.0.1", port="5000", debug=True)
