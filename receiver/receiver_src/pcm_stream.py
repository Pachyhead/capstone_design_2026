"""Parse the API's fixed 44-byte streaming WAV header across network packets."""
import struct


def iter_pcm(chunks):
    pending = bytearray()
    audio_format = None
    for chunk in chunks:
        pending.extend(chunk)
        if audio_format is None:
            if len(pending) < 44:
                continue
            header = pending[:44]
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE" or header[12:16] != b"fmt " or header[36:40] != b"data":
                raise ValueError("Expected the TTS API's PCM WAV header")
            size, encoding, channels, rate, byte_rate, align, bits = struct.unpack_from("<IHHIIHH", header, 16)
            if size != 16 or encoding != 1 or bits != 16 or channels != 1 or rate <= 0 or align != 2 or byte_rate != rate * align:
                raise ValueError("Expected mono PCM16 audio")
            audio_format = (rate, channels)
            del pending[:44]
        count = len(pending) // 2 * 2
        if count:
            pcm = bytes(pending[:count])
            del pending[:count]
            yield audio_format, pcm
    if audio_format is None or pending:
        raise ValueError("Truncated streaming WAV header or PCM sample")
