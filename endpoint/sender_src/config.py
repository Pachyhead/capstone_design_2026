import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 저장소 루트를 경로에 추가: grpc_getting_started, src/ 는 루트 한 곳에서만 관리한다
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
