import pandas as pd
import json
import http.client
import time


# 클로바 api 사용을 위한 세팅
class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/api-tools/embedding/v2/dac1ca79dd5d4a9d8e3b8770c74d94e0',
                     json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['embedding']
        elif res['status']['code'] == '42901':
            print('42901 error')
            time.sleep(20)
            return self.execute(completion_request)
        else:
            return 'Error'


completion_executor = CompletionExecutor(
    host='clovastudio.apigw.ntruss.com',
    api_key='api_key',
    api_key_primary_val='api_key_primary_val',
    request_id='request_id'
)

# 관광지 데이터 파일 열기
visit_area_info = pd.read_csv('basic_data/tn_visit_area_info_방문지정보_A.csv')
travel_info = pd.read_csv('basic_data/tn_travel_여행_A.csv')
traveler_info = pd.read_csv('basic_data/tn_traveller_master_여행객 Master_A.csv')

# 세 개의 파일에서 필요한 column 만으로 고름
travel_info = pd.concat([travel_info['TRAVEL_ID'],
                         travel_info['TRAVELER_ID']], axis=1)
# 관광객 정보 - 모델 입력
traveler_info = pd.concat([traveler_info['TRAVELER_ID'],
                           traveler_info['GENDER'],
                           traveler_info['AGE_GRP'],
                           traveler_info['TRAVEL_STATUS_ACCOMPANY'],
                           traveler_info['TRAVEL_STYL_1'],
                           traveler_info['TRAVEL_STYL_5'],
                           traveler_info['TRAVEL_STYL_6'],
                           traveler_info['TRAVEL_STYL_8']], axis=1)
# 관광지 정보 - 모델 입력, 관광지 평점 - 모델 출력
visit_area_info = pd.concat([visit_area_info['TRAVEL_ID'],
                             visit_area_info['VISIT_AREA_TYPE_CD'],
                             visit_area_info['VISIT_AREA_NM'],
                             visit_area_info['DGSTFN']], axis=1)

# 세 개의 파일을 'TRAVEL_ID', 'TRAVELER_ID' 을 기준으로 join
travel_info = pd.merge(travel_info, traveler_info, on='TRAVELER_ID')
travel_info.drop('TRAVELER_ID', axis=1, inplace=True)
visit_area_info = pd.merge(travel_info, visit_area_info, on='TRAVEL_ID')
visit_area_info.drop('TRAVEL_ID', axis=1, inplace=True)

# 방문지 파일에서 필요없는 행삭제
for i in visit_area_info.index:
    # 방문지 type 이 자연 관광지, 역사 / 유적 / 종교 시설, 레저 / 스포츠 관련 시설, 테마 시설, 둘레길, 지역 축제, 체험 활동 관광지가 아닌 방문지를 제거
    if visit_area_info['VISIT_AREA_TYPE_CD'][i] not in [1, 2, 5, 6, 7, 8, 13]:
        visit_area_info.drop(i, inplace=True)
    # 방문지 type 으로 필터링 할 수 없는 행제거 (리스트 추가 예정)
    elif visit_area_info['VISIT_AREA_NM'][i] in ['BHC치킨 에버랜드점', 'KFC 대학로점']:
        visit_area_info.drop(i, inplace=True)
    # 방문지 데이터의 일부 오류를 수정
    elif visit_area_info['VISIT_AREA_NM'][i] == '경북궁':
        visit_area_info.loc[i, 'VISIT_AREA_NM'] = '경복궁'

# 평점 결측치 제거
visit_area_info = visit_area_info.dropna(subset=['DGSTFN'])

# 방문지 이름을 기준으로 정렬
visit_area_info.sort_values(by='VISIT_AREA_NM', ascending=True, inplace=True)
visit_area_info.reset_index(drop=True, inplace=True)

# 관광지 목록 추출
place = pd.DataFrame(visit_area_info, columns=['VISIT_AREA_NM', 'VISIT_AREA_TYPE_CD'])
last_place = ''
for i in place.index:
    if place["VISIT_AREA_NM"][i] == last_place:
        place.drop(i, inplace=True)
    else:
        last_place = place["VISIT_AREA_NM"][i]
place.reset_index(drop=True, inplace=True)

# 목록의 관광지 이름을 임베딩
embedding = pd.DataFrame(columns=place['VISIT_AREA_NM'])
for i in place.index:
    visit_embedding = completion_executor.execute({"text": place['VISIT_AREA_NM'][i]})
    while visit_embedding == 'Error':
        visit_embedding = completion_executor.execute({"text": place['VISIT_AREA_NM'][i]})
    embedding[place['VISIT_AREA_NM'][i]] = pd.Series(visit_embedding, name=place['VISIT_AREA_NM'][i])
    print(i / len(place.index))

# 그외 정보 정수 인덱싱
visit_area_info.loc[visit_area_info['GENDER'] == '남', 'GENDER'] = 0
visit_area_info.loc[visit_area_info['GENDER'] == '여', 'GENDER'] = 1
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '나홀로 여행', 'TRAVEL_STATUS_ACCOMPANY'] = 0
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '2인 여행(가족 외)', 'TRAVEL_STATUS_ACCOMPANY'] = 1
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '2인 가족 여행', 'TRAVEL_STATUS_ACCOMPANY'] = 2
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '3인 이상 여행(가족 외)', 'TRAVEL_STATUS_ACCOMPANY'] = 3
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '부모 동반 여행', 'TRAVEL_STATUS_ACCOMPANY'] = 4
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '자녀 동반 여행', 'TRAVEL_STATUS_ACCOMPANY'] = 5
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '3인 동반 여행(친척 포함)', 'TRAVEL_STATUS_ACCOMPANY'] = 6
visit_area_info.loc[visit_area_info['TRAVEL_STATUS_ACCOMPANY'] == '3대 동반 여행(친척 포함)', 'TRAVEL_STATUS_ACCOMPANY'] = 7

# 전처리 파일을 저장
visit_area_info.to_csv("data_preprocessing/train.csv", index=True)
embedding.to_csv("data_preprocessing/place_name_to_embedding.csv", index=True)
place.to_csv("data_preprocessing/place_list.csv", index=True)
