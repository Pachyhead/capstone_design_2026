import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 저장소 루트를 경로에 추가: grpc_getting_started, src/ 는 루트 한 곳에서만 관리한다
# 맨 뒤에 추가해야 한다. 루트에도 receiver/, sender/ 디렉터리가 있어서 앞에 두면
# sender_src의 receiver.py, sender.py 대신 루트 패키지가 임포트된다.
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# 저장소 루트의 .env를 읽는다 (클라이언트 머신은 .env.client.example 형태). 이미 있는 값은 덮어쓰지 않는다.
load_dotenv(REPO_ROOT / ".env")
