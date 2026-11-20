"""Incremental 12 Hz codec decoding driven by a Talker forward hook.

The caller must serialize access to the shared model while the hook is installed.
This reuses the decoder's left-context chunking strategy, not a persistent KV cache.
"""


class FrameChunker:
    """Buffer new frames and retain context without emitting it twice."""

    def __init__(self, decode, emit, chunk_frames, context_frames=25, reference=()):
        if chunk_frames < 1 or context_frames < 0:
            raise ValueError("Invalid codec chunk/context size")
        self.decode, self.emit = decode, emit
        self.chunk_frames, self.context_frames = chunk_frames, context_frames
        self.context = list(reference)[-context_frames:] if context_frames else []
        self.pending = []

    def push(self, frame):
        self.pending.append(frame)
        if len(self.pending) >= self.chunk_frames:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        frames = self.context + self.pending
        audio = self.decode(frames, len(self.context))
        self.emit(audio)
        self.context = frames[-self.context_frames:] if self.context_frames else []
        self.pending = []


class CodecAudioHook:
    def __init__(self, model, callback, chunk_frames=4, context_frames=25,
                 reference=None, cancelled=None):
        import torch
        self.torch = torch
        self.model = model
        self.cancelled = cancelled or (lambda: False)
        self.eos = model.config.talker_config.codec_eos_token_id
        tokenizer = model.speech_tokenizer
        if tokenizer.get_model_type() != "qwen3_tts_tokenizer_12hz":
            raise ValueError("Live audio supports only the 12 Hz tokenizer")
        self.decoder = tokenizer.model.decoder
        self.sample_rate = tokenizer.get_output_sample_rate()
        self.upsample = tokenizer.get_decode_upsample_rate()
        self.groups = self.decoder.config.num_quantizers
        self.codebook_size = self.decoder.config.codebook_size
        self.stopped = False
        self.handle = None
        refs = [] if reference is None else list(reference[-context_frames:]) if context_frames else []
        self.chunker = FrameChunker(self._decode, lambda wav: callback(wav, self.sample_rate),
                                    chunk_frames, context_frames, refs)

    def _check_cancelled(self):
        if self.cancelled():
            raise InterruptedError("Audio generation cancelled")

    def _decode(self, frames, context_length):
        self._check_cancelled()
        torch = self.torch
        codes = torch.stack(frames).transpose(0, 1).unsqueeze(0)
        codes = codes.to(next(self.decoder.parameters()).device)
        with torch.inference_mode():
            wav = self.decoder(codes)[0, 0, context_length * self.upsample:]
        expected = (len(frames) - context_length) * self.upsample
        if wav.numel() != expected:
            raise RuntimeError(f"Decoder returned {wav.numel()} samples, expected {expected}")
        return wav.float().cpu().numpy()

    def _on_forward(self, module, args, output):
        self._check_cancelled()
        if self.stopped:
            return
        # Prefill has codec_ids=None. Later outputs contain a COMPLETE RVQ frame,
        # unlike codec_head logits, which describe only the next layer-0 token.
        codes = output.hidden_states[1]
        if codes is None:
            return
        if tuple(codes.shape) != (1, self.groups):
            raise ValueError("Live audio requires a single utterance and complete codec frames")
        if int(codes[0, 0]) == self.eos:
            self.stopped = True
            return
        if bool(((codes < 0) | (codes >= self.codebook_size)).any()):
            raise ValueError("Special or invalid token in audio codec frame")
        self.chunker.push(codes[0].detach())

    def __enter__(self):
        self.handle = self.model.talker.register_forward_hook(self._on_forward)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.handle.remove()
        if exc_type is None:
            self._check_cancelled()
            self.chunker.flush()
        return False
