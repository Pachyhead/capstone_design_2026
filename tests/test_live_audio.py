"""Run with: python -m unittest discover -s tests -p test_live_audio.py -v"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import struct
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = load("live_audio_test_module", "qwen_tts/inference/live_audio.py")
worker_module = load("stream_worker_test_module", "qwen_tts_api/stream_worker.py")
pcm_module = load("pcm_stream_test_module", "receiver/receiver_src/pcm_stream.py")


class ChunkTests(unittest.TestCase):
    def test_reference_context_not_replayed_and_partial_tail_flushed(self):
        emitted, calls = [], []
        def decode(frames, context):
            calls.append((list(frames), context))
            return frames[context:]
        c = live.FrameChunker(decode, emitted.append, 4, 2, [-3, -2, -1])
        for frame in range(7):
            c.push(frame)
        self.assertEqual(emitted, [[0, 1, 2, 3]])  # before generation finishes
        c.flush()
        c.flush()
        self.assertEqual(emitted, [[0, 1, 2, 3], [4, 5, 6]])
        self.assertEqual(calls, [([-2, -1, 0, 1, 2, 3], 2), ([2, 3, 4, 5, 6], 2)])

    def test_empty_and_short_generation(self):
        out = []
        c = live.FrameChunker(lambda frames, context: frames[context:], out.append, 4, 0)
        c.flush()
        self.assertEqual(out, [])
        c.push(1)
        c.flush()
        self.assertEqual(out, [[1]])


class WorkerTests(unittest.TestCase):
    def test_first_audio_precedes_producer_completion(self):
        finish = threading.Event()
        def produce(emit, cancelled):
            emit(b"first")
            finish.wait(2)
            emit(b"tail")
        w = worker_module.AudioWorker(produce).start()
        try:
            self.assertEqual(w.read(), b"first")
            self.assertFalse(w.done.is_set())
            finish.set()
            self.assertEqual(w.read(), b"tail")
            self.assertIsNone(w.read())
        finally:
            finish.set()
            w.close()
            w.thread.join(2)

    def test_disconnect_unblocks_full_queue(self):
        def produce(emit, cancelled):
            while True:
                emit(b"audio")
        w = worker_module.AudioWorker(produce, capacity=1).start()
        w.close()
        w.thread.join(2)
        self.assertFalse(w.thread.is_alive())

    def test_failure_before_and_after_audio_is_propagated(self):
        for prefix in (False, True):
            def produce(emit, cancelled):
                if prefix:
                    emit(b"first")
                raise ValueError("decoder failed")
            w = worker_module.AudioWorker(produce).start()
            if prefix:
                self.assertEqual(w.read(), b"first")
            with self.assertRaisesRegex(ValueError, "decoder failed"):
                w.read()
            w.thread.join(2)


class PCMTests(unittest.TestCase):
    @staticmethod
    def wav():
        header = (b"RIFF" + struct.pack("<I", 0xffffffff) + b"WAVEfmt "
                  + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
                  + b"data" + struct.pack("<I", 0xffffffff))
        return header + struct.pack("<hhhh", 1, -2, 3, -4)

    def test_arbitrary_network_boundaries(self):
        wav = self.wav()
        for size in (1, 3, 17, 44, 100):
            out = list(pcm_module.iter_pcm(wav[i:i+size] for i in range(0, len(wav), size)))
            self.assertEqual(b"".join(p for _, p in out), wav[44:])
            self.assertTrue(all(fmt == (24000, 1) for fmt, _ in out))

    def test_truncation(self):
        for wav in (self.wav()[:20], self.wav()[:-1]):
            with self.assertRaises(ValueError):
                list(pcm_module.iter_pcm([wav]))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable; run these hook tests on the TTS server")
class TorchHookTests(unittest.TestCase):
    def make_model(self):
        class Decoder(torch.nn.Module):
            config = SimpleNamespace(num_quantizers=2, codebook_size=100)
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
            def forward(self, codes):
                return codes[:, :1].float().repeat_interleave(2, dim=-1)
        class Talker(torch.nn.Module):
            def forward(self, codes):
                return SimpleNamespace(hidden_states=(None, codes))
        tokenizer = SimpleNamespace(
            model=SimpleNamespace(decoder=Decoder()),
            get_model_type=lambda: "qwen3_tts_tokenizer_12hz",
            get_output_sample_rate=lambda: 24000,
            get_decode_upsample_rate=lambda: 2,
        )
        return SimpleNamespace(talker=Talker(), speech_tokenizer=tokenizer,
                               config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=101)))

    def test_real_hook_prefill_frames_eos_and_tail(self):
        m, out = self.make_model(), []
        ref = torch.tensor([[80, 80], [90, 90]])
        with live.CodecAudioHook(m, lambda wav, sr: out.extend(wav.tolist()), 2, 2, ref):
            m.talker(None)
            self.assertEqual(out, [])
            m.talker(torch.tensor([[1, 2]]))
            m.talker(torch.tensor([[3, 4]]))
            self.assertEqual(out, [1, 1, 3, 3])
            m.talker(torch.tensor([[5, 6]]))
            m.talker(torch.tensor([[101, 0]]))
        self.assertEqual(out, [1, 1, 3, 3, 5, 5])
        self.assertEqual(len(m.talker._forward_hooks), 0)

    def test_exception_and_cancellation_remove_hook(self):
        m = self.make_model()
        for cancel in (False, True):
            with self.assertRaises((RuntimeError, InterruptedError)):
                with live.CodecAudioHook(m, lambda *_: None, cancelled=lambda: cancel):
                    m.talker(None)
                    raise RuntimeError("failed")
            self.assertEqual(len(m.talker._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
