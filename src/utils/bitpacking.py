from bitarray import bitarray # 비트 조작 라이브러리
from bitarray.util import int2ba, ba2int

def encode_packet(send_id: int, receiver_id: int, emo_type: int, emotion_indices: list[int], message: str) -> bytes:
    """
    비트패킹: [프레임 크기(16bits)] [헤더(40bits)] [메시지(Variable)]
    헤더: [send_id(6bits)] [receiver_id(6bits)] [emo_type(4bits)] [emotion_vec(24bits)]
    """
    header = bitarray(40) # 헤더 크기 5bytes
    header.setall(False) # 비트 0으로 초기화

    header[0:6] = int2ba(send_id, length=6)
    header[6:12] = int2ba(receiver_id, length=6)
    header[12:16] = int2ba(emo_type, length=4)

    emo_pck = bitarray()
    bit_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
    for val, size in zip(emotion_indices, bit_sizes):
        emo_pck.extend(int2ba(val, length=size))
    header[16:40] = emo_pck # 감정벡터 크기 24bits

    message_bytes = message.encode('utf-8') # 메세지를 bytes로 변환

    frame_size = 5 + len(message_bytes) # 헤더 크기 + 메시지 크기

    packet = bitarray()
    packet.extend(int2ba(frame_size, length=16)) # 프레임 크기 계산 후, 앞에 추가(2byte 추가됨)
    packet.extend(header)

    message_bits = bitarray()
    message_bits.frombytes(message_bytes) # bytes를 bitarray로 변환
    packet.extend(message_bits)

    return packet.tobytes()

def decode_packet(data: bytes) -> tuple:
    """비트패킹 패킷 역해석"""
    packet = bitarray()
    packet.frombytes(data) # bitarray 형태로 변환

    frame_size = ba2int(packet[0:16]) # 길이 정보
    header = packet[16:40+16] # 헤더만 가져옴
    message_bits = packet[40+16:frame_size * 8 + 16] # 메시지만 가져옴. 프레임 크기 + 16비트(2바이트)

    send_id = ba2int(header[0:6])
    receiver_id = ba2int(header[6:12])
    emo_type = ba2int(header[12:16])

    emo_pck = header[16:40]
    emotion_indices = []
    for i in range(0, len(emo_pck), 3):
        emotion_indices.append(ba2int(emo_pck[i:i+3]))

    message_bytes = message_bits.tobytes()
    message = message_bytes.decode('utf-8') # 메시지 다시 문자열로 되돌림

    return (send_id, receiver_id, emo_type, emotion_indices, message)