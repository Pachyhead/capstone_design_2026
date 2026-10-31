#!/usr/bin/env bash
# 목 서버(mock_relay.py) 테스트용 사설 CA와 서버 인증서를 만든다.
# GPU 서버의 인증서와 무관한 테스트 전용 체인이며, 결과물은 git에 올리지 않는다.
#
#   cd endpoint && bash gen_mock_certs.sh
#
# 엔드포인트 .env에는 CA_CERT_PATH=endpoint/mock_certs/ca.pem 을 넣는다.
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash가 -subj의 /CN= 값을 경로로 바꾸는 것을 막는다

OUT="$(cd "$(dirname "$0")" && pwd)/mock_certs"
MOCK_IP="${MOCK_IP:-127.0.0.1}"
mkdir -p "$OUT" && cd "$OUT"

openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 365 \
  -subj "/CN=capstone-mock-ca"

openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr \
  -subj "/CN=${MOCK_IP}"

printf "subjectAltName = IP:%s\nextendedKeyUsage = serverAuth\n" "$MOCK_IP" > server.ext

openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial \
  -out server.pem -days 365 -extfile server.ext

openssl verify -CAfile ca.pem server.pem
echo "generated in $OUT"
