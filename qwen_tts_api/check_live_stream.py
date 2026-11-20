"""Measure first PCM arrival (excluding the WAV header) on a running API."""
import argparse
import json
import struct
import time
import wave
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080/stream")
    parser.add_argument("--payload", required=True, help="Request JSON with server-visible paths")
    parser.add_argument("--output", default="live.wav")
    args = parser.parse_args()
    with open(args.payload, encoding="utf-8") as f:
        payload = json.load(f)
    start = time.perf_counter()
    first = None
    total = 0
    pending = bytearray()
    rate = None
    with requests.post(args.url, json=payload, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        with wave.open(args.output, "wb") as target:
            for chunk in response.iter_content(chunk_size=None):
                pending.extend(chunk)
                if rate is None:
                    if len(pending) < 44:
                        continue
                    if pending[:4] != b"RIFF" or pending[36:40] != b"data":
                        raise ValueError("Unexpected WAV header")
                    rate = struct.unpack_from("<I", pending, 24)[0]
                    target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                    del pending[:44]
                size = len(pending) // 2 * 2
                if size:
                    if first is None:
                        first = time.perf_counter() - start
                        print(f"First PCM: {first:.3f}s", flush=True)
                    target.writeframesraw(bytes(pending[:size]))
                    total += size // 2
                    del pending[:size]
    elapsed = time.perf_counter() - start
    if first is None or pending:
        raise RuntimeError("Empty or truncated PCM response")
    print(f"Complete: {elapsed:.3f}s; audio: {total/rate:.3f}s; samples: {total}")
    print(f"First-PCM lead over completion: {elapsed-first:.3f}s")


if __name__ == "__main__":
    main()
