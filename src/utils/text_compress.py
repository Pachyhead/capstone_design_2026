"""한글 텍스트 전송 인코딩

패킷: byte 0 = 플래그 (0: UTF-8, 1: cp949), byte 1~ = 인코딩된 텍스트.
"""

KOREAN_CODEC = "cp949"
M_UTF8 = 0
M_CP949 = 1

def compress_text(text: str) -> bytes:
    """텍스트 -> [flag][encoded]. 결과는 항상 UTF-8 원문 + 1B 이하."""
    try:
        return bytes([M_CP949]) + text.encode(KOREAN_CODEC)
    except UnicodeEncodeError:
        return bytes([M_UTF8]) + text.encode("utf-8")

def decompress_text(blob: bytes) -> str:
    """[flag][encoded] -> 텍스트."""
    if not blob:
        return ""
    method, payload = blob[0], blob[1:]
    if method == M_CP949:
        return payload.decode(KOREAN_CODEC)
    if method == M_UTF8:
        return payload.decode("utf-8")
    raise ValueError(f"알 수 없는 인코딩 플래그: {method}")
