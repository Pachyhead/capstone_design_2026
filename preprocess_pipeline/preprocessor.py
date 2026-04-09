from .emotion import EmotionExtractor
from .stt import WhisperSTT
from .vad import VadSegmenter
from .config import Config
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf

class Preprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.stt = WhisperSTT(config)
        self.emotion = EmotionExtractor(config)
        self._segmenter = None
        self.executor = ThreadPoolExecutor(max_workers=2)

    @property
    def segmenter(self) -> VadSegmenter:
        if self._segmenter is None:
            self._segmenter = VadSegmenter(self.config)
        return self._segmenter

    def process_chunk(
            self,
            audio_data: np.ndarray, 
            sr: int
        ) -> tuple[str, np.ndarray, list[str], list[float]]:
        """
        실시간 파이프라인 용
        
        Args:        
        - audio_data (np.ndarray): 1D numpy 오디오 데이터
        - sr (int): 샘플링 레이트 (Hz)
        
        Returns:        
        - tuple[str, np.ndarray, list[str], list[float]]: (STT 텍스트, 감정 임베딩, 감정 레이블, 감정 점수)
        """

        # 병렬 처리: STT와 감정 추출 동시에 실행
        future_text = self.executor.submit(self.stt.transcribe, audio_data)
        future_emotion = self.executor.submit(self.emotion.extract, audio_data, sr)
        text = future_text.result()
        emotion_vec, emotion_labels, emotion_scores = future_emotion.result()

        return text, emotion_vec, emotion_labels, emotion_scores

    def process_file(
            self, 
            audio_path: str, 
            output_dir: Path,
            global_counter: list,
    ) -> list[dict]:
        """
        데이터셋 생성용 (단일 오디오 파일 처리 to 메타데이터 리스트 반환)
        
        Args:
        - audio_path: 오디오 파일 경로
        - output_dir: 출력 디렉토리 Path 객체
        - global_counter: 전체 발화에 대한 글로벌 카운터 (mutable list로 전달)
        
        Returns:
        - list[dict]: 각 발화 청크의 메타데이터 리스트
        """

        # 원본 파일 정보
        source_name = Path(audio_path).stem
        
        # VAD 세그멘팅
        chunks = self.segmenter.segment(audio_path)

        results = []
        for chunk in chunks:
            chunk_id = f"{global_counter[0]:06d}"
            global_counter[0] += 1

            audio_data = chunk["audio"]
            sr = chunk["sr"]

            text, emotion_vec, emotion_labels, emotion_scores = self.process_chunk(chunk["audio"], chunk["sr"])
            
            if not text:
                text = ""
                meta["non_verbal"] = True

            # 청크 오디오 저장
            chunk_path = output_dir / "chunks" / f"{chunk_id}.wav"
            sf.write(str(chunk_path), audio_data, sr)
 
            # 감정 벡터 저장 (별도 npy)
            emb_path = output_dir / "embeddings" / f"{chunk_id}.npy"
            np.save(str(emb_path), emotion_vec)
            
            # 메타데이터
            meta = {
                "id": chunk_id,
                "source_file": source_name,
                "wav": f"chunks/{chunk_id}.wav",
                "text": text,
                "emo_labels": emotion_labels,
                "emo_scores": [float(s) for s in emotion_scores],
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "duration_sec": round(chunk["end_sec"] - chunk["start_sec"], 3),
            }
            results.append(meta)
            print(f"[{chunk_id}] {meta['duration_sec']}s | \"{meta['text'][:30]}...\"")

        return results