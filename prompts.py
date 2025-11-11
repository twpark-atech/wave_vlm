# prompts.py
import json

CATEGORIES = [
    "의료직","요식/조리","건설/현장","물류/배송","제조/공장","과학/실험","공공안전/치안",
    "군/경비","교육/보육","사무/기업","리테일/판매","미용/뷰티","스포츠/피트니스",
    "관광/항공/호텔","예술/공연/미디어","기술/IT/크리에이터","건물관리/청소",
    "농림어업/야외","학생/교복","종교/성직","운수/대중교통"
]

# 각 직업군별 의복/장비 특징 사전 (프롬프트 힌트용)
CATEGORY_HINTS = {
    "의료직": {
        "keywords": ["스크럽", "화이트코트", "청진기", "헤어캡", "의료마스크", "클로그/크록스", "네임택"],
        "strong_cues": ["스크럽 세트", "청진기", "흰 가운"],
        "disambiguation": "과학/실험과 구분: 보안경/니트릴장갑 대신 청진기·스크럽 여부 확인"
    },
    "요식/조리": {
        "keywords": ["셰프모", "셰프재킷(더블브레스트)", "앞치마(데님/가죽)", "주방장갑", "미끄럼방지화"],
        "strong_cues": ["흰 셰프재킷", "토크(셰프모)", "두꺼운 앞치마"],
        "disambiguation": "바리스타/서빙과 구분: 조리복 상의(재킷)·토크 유무 확인"
    },
    "건설/현장": {
        "keywords": ["형광 안전조끼(반사띠)", "하드햇", "카고 작업바지", "안전화", "장갑", "무전기"],
        "strong_cues": ["하이비즈 조끼", "하드햇"],
        "disambiguation": "제조/공장과 구분: 반사띠·하드햇 비중↑(야외), 제조는 실내용 작업복"
    },
    "물류/배송": {
        "keywords": ["상자/박스", "핸드트럭", "스캐너/바코드리더", "기능성 폴로", "경량점퍼", "캡모자", "장갑"],
        "strong_cues": ["스캐너", "박스 취급"],
        "disambiguation": "리테일과 구분: 매장 유니폼보다 야외 이동 장비·상자 노출"
    },
    "제조/공장": {
        "keywords": ["점프수트", "작업복 상·하 세트", "산업용 안전모", "귀마개", "장갑", "안전화", "명찰 패치"],
        "strong_cues": ["동일색 작업복 세트", "명찰 패치"],
        "disambiguation": "건설/현장과 구분: 반사띠↓, 실내용 작업복 느낌"
    },
    "과학/실험": {
        "keywords": ["랩코트", "보안경", "니트릴 장갑", "샘플 튜브", "클립보드"],
        "strong_cues": ["보안경", "흰 가운(청진기 없음)"],
        "disambiguation": "의료직과 구분: 보안경/장갑 비중↑, 청진기 없음"
    },
    "공공안전/치안": {
        "keywords": ["제복 셔츠", "계급장/라펠핀", "다용도 벨트", "수갑", "무전기", "네임패치"],
        "strong_cues": ["다용도 벨트", "제복형 셔츠/패치"],
        "disambiguation": "군/경비와 구분: 카모플라주↓, 도시형 제복과 계급장 강조"
    },
    "군/경비": {
        "keywords": ["카모플라주", "전술조끼", "컴뱃부츠", "베레/전술캡", "패치"],
        "strong_cues": ["군복무늬", "전술장구"],
        "disambiguation": "공공안전과 구분: 야전/전술 요소(카모, 전술조끼) 강조"
    },
    "교육/보육": {
        "keywords": ["니트/가디건", "편한 슬랙스", "네임택(학교/유치원)", "호루라기", "교구"],
        "strong_cues": ["단정 캐주얼", "교육기관 네임택"],
        "disambiguation": "사무/기업과 구분: 수트/블레이저 비중↓, 활동성↑"
    },
    "사무/기업": {
        "keywords": ["수트/블레이저", "드레스셔츠", "넥타이", "로퍼/드레스슈즈", "아이디카드 홀더"],
        "strong_cues": ["수트 셋업", "블레이저", "목걸이 ID 카드"],
        "disambiguation": "관광/항공/호텔과 구분: 승무원 스카프/특정 유니폼 규격 유무 확인"
    },
    "리테일/판매": {
        "keywords": ["매장 유니폼(폴로/셔츠)", "네임태그", "무전기", "포스기 주변 소품", "스니커즈"],
        "strong_cues": ["브랜드 유니폼", "네임태그"],
        "disambiguation": "물류/배송과 구분: 매장 환경 유니폼 vs 야외 이동 장비"
    },
    "미용/뷰티": {
        "keywords": ["블랙 앞치마/튜닉", "미용가위", "빗", "도구 파우치", "위생장갑", "마스크"],
        "strong_cues": ["블랙 앞치마", "헤어 도구 파우치"],
        "disambiguation": "요식 앞치마와 혼동 시 가위·빗 등 미용 도구 확인"
    },
    "스포츠/피트니스": {
        "keywords": ["저지", "트레이닝복/트랙수트", "레깅스", "휘슬", "스톱워치", "코치 클립보드", "운동화"],
        "strong_cues": ["저지/트랙수트", "코치 소품"],
        "disambiguation": "기술/크리에이터 캐주얼과 구분: 스포츠 로고/등번호 등 경기성 요소"
    },
    "관광/항공/호텔": {
        "keywords": ["승무원 유니폼", "스카프", "모자", "호텔리어 수트", "네임태그", "장갑(세레모니)"],
        "strong_cues": ["승무원 스카프", "정돈된 유니폼"],
        "disambiguation": "사무/기업과 구분: 항공/호텔 특유의 액세서리·규격"
    },
    "예술/공연/미디어": {
        "keywords": ["올블랙 스태프복", "헤드셋/인터컴", "카메라", "붐마이크", "무대 의상"],
        "strong_cues": ["헤드셋/카메라 장비", "의도적 무대 의상"],
        "disambiguation": "공공안전/보안과 구분: 촬영/오디오 장비 여부"
    },
    "기술/IT/크리에이터": {
        "keywords": ["후디/티셔츠", "캐주얼", "백팩", "헤드셋/마이크", "웹캠", "랩탑 소품"],
        "strong_cues": ["후디/캐주얼", "콘텐츠 제작 장비"],
        "disambiguation": "사무/기업과 구분: 격식↓, 장비 중심 캐주얼↑"
    },
    "건물관리/청소": {
        "keywords": ["작업복/베스트", "고무장갑", "청소 앞치마", "밀대", "세정도구", "캡모자"],
        "strong_cues": ["청소 도구 휴대", "실무형 유니폼"],
        "disambiguation": "제조/공장과 구분: 청소 도구 노출 여부"
    },
    "농림어업/야외": {
        "keywords": ["야상/방수점퍼", "넓은챙 모자", "장화", "멜빵바지", "작업장갑"],
        "strong_cues": ["장화", "방수의류/모자"],
        "disambiguation": "건설/현장과 구분: 반사띠/하드햇 유무"
    },
    "학생/교복": {
        "keywords": ["교복 블레이저", "셔츠", "넥타이/리본", "체크스커트/슬랙스", "학교 배지"],
        "strong_cues": ["교복 세트", "학교 배지"],
        "disambiguation": "사무 정장과 구분: 체크/교표/타이 형태 등 교복 특징"
    },
    "종교/성직": {
        "keywords": ["수단/가사", "성직복", "성직 모자", "십자목걸이", "전례용 스톨"],
        "strong_cues": ["성직복 실루엣", "종교 상징물"],
        "disambiguation": "공연 의상과 혼동 시 상징물 확인"
    },
    "운수/대중교통": {
        "keywords": ["운수사 제복", "캡/모자", "타이", "에폭시 패치", "장갑", "티켓장/펀치기(구형)", "네임태그"],
        "strong_cues": ["운수 유니폼", "캡/타이"],
        "disambiguation": "공공안전과 구분: 교통사 로고/티켓 장비 성격"
    }
}

HINTS_JSON = json.dumps(CATEGORY_HINTS, ensure_ascii=False, indent=2)

# 스키마 예시는 반드시 '유효한 JSON 숫자'로 제시 (범위 표기 금지)
SCHEMA = r"""
{
  "top_prediction": {"job": "직업명", "confidence": 0.83},
  "candidates": [
    {"job": "의료직", "prob": 0.01},
    {"job": "요식/조리", "prob": 0.01},
    {"job": "건설/현장", "prob": 0.02},
    {"job": "물류/배송", "prob": 0.02},
    {"job": "제조/공장", "prob": 0.01},
    {"job": "과학/실험", "prob": 0.01},
    {"job": "공공안전/치안", "prob": 0.01},
    {"job": "군/경비", "prob": 0.01},
    {"job": "교육/보육", "prob": 0.03},
    {"job": "사무/기업", "prob": 0.60},
    {"job": "리테일/판매", "prob": 0.05},
    {"job": "미용/뷰티", "prob": 0.01},
    {"job": "스포츠/피트니스", "prob": 0.02},
    {"job": "관광/항공/호텔", "prob": 0.02},
    {"job": "예술/공연/미디어", "prob": 0.01},
    {"job": "기술/IT/크리에이터", "prob": 0.08},
    {"job": "건물관리/청소", "prob": 0.01},
    {"job": "농림어업/야외", "prob": 0.01},
    {"job": "학생/교복", "prob": 0.02},
    {"job": "종교/성직", "prob": 0.01},
    {"job": "운수/대중교통", "prob": 0.01}
  ],
  "evidence": ["셔츠","재킷","백팩"]
}
""".strip()

PROMPT_RULES = """
규칙:
1) 아래 21개 카테고리만 사용하고, 철자/띄어쓰기를 정확히 맞추세요. 새 라벨을 만들지 마세요.
2) 아래 21개 카테고리 중에서 상위 3개만 선택하여 candidates 배열에 넣으세요. (정확히 3개)
3) 확신이 낮더라도 소수의 상위 후보에만 과도하게 몰지 말고, 나머지에도 소량의 꼬리 확률을 유지하세요.
4) 각 prob는 최소 0.005 이상, 최대 0.90 이하로 설정하세요. (어떤 클래스도 1.0 금지)
5) top_prediction.confidence는 해당 top 후보의 prob와 동일하게 하세요.
6) 출력은 반드시 순수 JSON만 작성하세요(설명/마크다운/코드블록 금지).
7) 아래 'CUE_BOOK'의 키워드를 참조하여 evidence 배열을 작성하세요. evidence는 감지된 의복/장비 단서 문자열 배열입니다.
""".strip()

EXAMPLE = r"""
예시:
{
  "top_prediction": {"job": "과학/실험", "confidence": 0.60},
  "candidates": [
    {"job":"의료직","prob":0.01},{"job":"요식/조리","prob":0.01},{"job":"건설/현장","prob":0.02},
    {"job":"물류/배송","prob":0.02},{"job":"제조/공장","prob":0.01},{"job":"사무/기업","prob":0.01},
    {"job":"공공안전/치안","prob":0.01},{"job":"군/경비","prob":0.01},{"job":"교육/보육","prob":0.03},
    {"job":"과학/실험","prob":0.60},{"job":"리테일/판매","prob":0.05},{"job":"미용/뷰티","prob":0.01},
    {"job":"스포츠/피트니스","prob":0.02},{"job":"관광/항공/호텔","prob":0.02},{"job":"예술/공연/미디어","prob":0.01},
    {"job":"기술/IT/크리에이터","prob":0.08},{"job":"건물관리/청소","prob":0.01},{"job":"농림어업/야외","prob":0.01},
    {"job":"학생/교복","prob":0.02},{"job":"종교/성직","prob":0.01},{"job":"운수/대중교통","prob":0.01}
  ],
  "evidence":["셔츠","노트북","백팩"]
}
""".strip()

def get_vlm_prompt(lang: str = "ko") -> str:
    # 모델이 evidence를 더 잘 뽑도록 카테고리 힌트(CUE_BOOK)를 함께 제공
    return (
        "아래 이미지는 배경이 제거된 사람의 의상(누끼)입니다.\n"
        "의상 스타일과 소지품 단서를 바탕으로 예상 직업을 추정해 주세요.\n"
        "반드시 JSON으로만 응답하세요.\n\n"
        + "CATEGORIES:\n" + json.dumps(CATEGORIES, ensure_ascii=False) + "\n\n"
        + "CUE_BOOK:\n" + HINTS_JSON + "\n\n"
        + "SCHEMA:\n" + SCHEMA + "\n\n"
        + PROMPT_RULES + "\n\n"
        + EXAMPLE
    )

def get_categories(lang: str = "ko"):
    return CATEGORIES

def get_system_prompt(lang: str = "ko") -> str:
    return (
        "당신은 한국어만 사용하는 도우미입니다. "
        "반드시 한국어 JSON만 출력하고, 다른 언어(중국어/영어 등)는 절대 사용하지 마세요. "
        "키 이름과 직업 카테고리 명칭도 전부 한국어를 사용하세요. "
        "설명 문장이나 주석 없이 JSON만 출력하세요."
    )

def reduce_to_top3(parsed: dict) -> dict:
    """
    parsed(JSON dict)에서 candidates를 prob 내림차순 상위 3개만 남기고,
    top_prediction과 confidence를 candidates[0] 기준으로 재설정한다.
    prob 합이 1.0이 되도록 재정규화한다.
    """
    try:
        cands = parsed.get("candidates") or []
        # 숫자형 prob만 남기고 정렬
        clean = []
        for it in cands:
            job = it.get("job")
            p = it.get("prob", 0)
            try:
                pv = float(str(p).strip().rstrip("%"))  # 문자열이면 처리
            except Exception:
                pv = 0.0
            # 퍼센트 표기 방지: 1.0 초과면 0~1 스케일로 가정하지 않고 그대로 둔다
            if pv > 1.0:
                pv = pv / 100.0
            clean.append({"job": job, "prob": max(0.0, pv)})

        clean.sort(key=lambda x: x["prob"], reverse=True)
        top3 = clean[:3] if len(clean) >= 3 else clean

        s = sum(x["prob"] for x in top3) or 1.0
        top3 = [{"job": x["job"], "prob": (x["prob"] / s)} for x in top3]

        if top3:
            parsed["top_prediction"] = {"job": top3[0]["job"], "confidence": top3[0]["prob"]}
        parsed["candidates"] = top3
        return parsed
    except Exception:
        return parsed