"""저장소 루트의 .env를 읽는다.

서버와 클라이언트 모두 저장소 루트의 .env 하나만 쓴다.
- 서버 머신: .env.server.example을 복사해서 루트 .env로 사용
- 클라이언트 머신: .env.client.example을 복사해서 루트 .env로 사용
"""

from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env() -> None:
    """루트 .env를 환경변수로 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다."""
    load_dotenv(ENV_PATH)
