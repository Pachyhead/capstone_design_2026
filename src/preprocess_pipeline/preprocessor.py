from preprocess_pipeline.emotion import EmotionExtractor
from preprocess_pipeline.stt import WhisperSTT
from concurrent.futures import ThreadPoolExecutor
from preprocess_pipeline.vad import VadSegmenter
from preprocess_pipeline.config import Config
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import soundfile as sf

class Preprocessor:
    def __init__(self, config: Config, executor: ThreadPoolExecutor):
        self.config = config
        self.stt = WhisperSTT(config)
        self.emotion = EmotionExtractor(config)
        self._segmenter = None
        self.executor = executor

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
            start_id: int,
    ) -> tuple[list[dict], int]:
        """
        데이터셋 생성용 (단일 오디오 파일 처리 to 메타데이터 리스트 반환)

        Args:
        - audio_path: 오디오 파일 경로
        - output_dir: 출력 디렉토리 Path 객체
        - start_id: 이 파일의 첫 청크에 부여할 id

        Returns:
        - tuple[list[dict], int]: (각 발화 청크의 메타데이터 리스트, 다음에 사용할 id)
        """

        # 원본 파일 정보
        source_name = Path(audio_path).stem

        # VAD 세그멘팅
        chunks = self.segmenter.segment(audio_path)

        results = []
        next_id = start_id
        for chunk in chunks:
            chunk_id = f"{next_id:06d}"
            next_id += 1

            audio_data = chunk["audio"]
            sr = chunk["sr"]

            text, emotion_vec, emotion_labels, emotion_scores = self.process_chunk(audio_data, sr)

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
                "non_verbal": (text.strip() == ""),
                "emo_vec": f"embeddings/{chunk_id}.npy",
                "emo_labels": emotion_labels,
                "emo_scores": [float(s) for s in emotion_scores],
                "start_sec": chunk["start_sec"],
                "end_sec": chunk["end_sec"],
                "duration_sec": round(chunk["end_sec"] - chunk["start_sec"], 3),
            }
            results.append(meta)
            tqdm.write(f"[{chunk_id}] {meta['duration_sec']}s | \"{text[:30]}...\"")

        return results, next_id