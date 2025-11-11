# 직업 예측 AI
## 사용 방법
### 1. 가상환경 생성
```bash
python -m venv .venv
```

### 2. 가상환경 실행
```bash
source .venv/bin/actiavte
```

### 3. 필수 구성요소 설치
```bash
pip install -r requirements.txt
```

### 4. 서버 실행
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. 페이지 접속
url: http://localhost:8000