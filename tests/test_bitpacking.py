import pytest
from src.utils.bitpacking import encode_packet, decode_packet


class TestBitpacking:
    """비트패킹 단위 테스트"""

    def test_encode_packet_basic(self):
        """기본 패킷 인코딩 테스트"""
        send_id = 1
        receiver_id = 2
        emo_type = 3
        emotion_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        message = "Hello"

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)

        assert isinstance(packet, bytes)
        assert len(packet) > 0

    def test_decode_packet_basic(self):
        """기본 패킷 디코딩 테스트"""
        send_id = 1
        receiver_id = 2
        emo_type = 3
        emotion_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        message = "Hello"

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)
        decoded = decode_packet(packet)

        assert decoded[0] == send_id  # send_id
        assert decoded[1] == receiver_id  # receiver_id
        assert decoded[2] == emo_type  # emo_type
        assert decoded[3] == emotion_indices  # emotion_indices
        assert decoded[4] == message  # message

    def test_round_trip_empty_message(self):
        """빈 메시지 라운드트립 테스트"""
        send_id = 0
        receiver_id = 3
        emo_type = 0
        emotion_indices = [0, 0, 0, 0, 0, 0, 0, 0]
        message = ""

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)
        decoded = decode_packet(packet)

        assert decoded[0] == send_id
        assert decoded[1] == receiver_id
        assert decoded[2] == emo_type
        assert decoded[3] == emotion_indices
        assert decoded[4] == message

    def test_round_trip_long_message(self):
        """긴 메시지 라운드트립 테스트"""
        send_id = 2
        receiver_id = 1
        emo_type = 2
        emotion_indices = [7, 6, 5, 4, 3, 2, 1, 0]
        message = "This is a longer message with multiple words and sentences. " * 10

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)
        decoded = decode_packet(packet)

        assert decoded[0] == send_id
        assert decoded[1] == receiver_id
        assert decoded[2] == emo_type
        assert decoded[3] == emotion_indices
        assert decoded[4] == message

    def test_round_trip_korean_message(self):
        """한글 메시지 라운드트립 테스트 (UTF-8 인코딩)"""
        send_id = 3
        receiver_id = 0
        emo_type = 4
        emotion_indices = [1, 2, 3, 4, 5, 6, 7, 0]
        message = "안녕하세요. 이것은 한글 메시지입니다."

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)
        decoded = decode_packet(packet)

        assert decoded[0] == send_id
        assert decoded[1] == receiver_id
        assert decoded[2] == emo_type
        assert decoded[3] == emotion_indices
        assert decoded[4] == message

    def test_special_characters_in_message(self):
        """메시지에 특수문자 포함 테스트"""
        send_id = 1
        receiver_id = 2
        emo_type = 1
        emotion_indices = [4, 4, 4, 4, 4, 4, 4, 4]
        message = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)
        decoded = decode_packet(packet)

        assert decoded[4] == message

    def test_newline_in_message(self):
        """메시지에 줄바꿈 포함 테스트"""
        send_id = 0
        receiver_id = 1
        emo_type = 3
        emotion_indices = [2, 2, 2, 2, 2, 2, 2, 2]
        message = "Line 1\nLine 2\nLine 3"

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)
        decoded = decode_packet(packet)

        assert decoded[4] == message

    def test_packet_size_consistency(self):
        """패킷 크기 일관성 테스트"""
        send_id = 1
        receiver_id = 2
        emo_type = 1
        emotion_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        message = "Test"

        packet = encode_packet(send_id, receiver_id, emo_type, emotion_indices, message)

        # 프레임 크기 = 헤더(5) + 메시지
        frame_size = int.from_bytes(packet[0:2], byteorder='big')

        # 실제 패킷 크기 (프레임 크기는 헤더+메시지만, 프레임 크기 필드 자체는 포함 안 함)
        expected_size = frame_size + 2  # +2는 프레임 크기 필드 자체
        assert len(packet) == expected_size