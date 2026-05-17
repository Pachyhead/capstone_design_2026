import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
from preprocess_pipeline.config import Config
from preprocess_pipeline.preprocessor import Preprocessor

def load_processed_state(metadata_path: Path) -> tuple[set[str], int]:
    """
    기존 jsonl에서 처리완료 source_file 집합과 다음 chunk_id를 복원
    """
    processed_sources: set[str] = set()
    max_id = -1

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    meta = json.loads(line)
                except json.JSONDecodeError:
                    continue
                processed_sources.add(meta["source_file"])
                max_id = max(max_id, int(meta["id"]))
    return processed_sources, max_id + 1

def make_dir(dir: str):
    """
    출력 디렉토리 생성
    """
    output_dir = Path(dir)
    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    error_log_path = output_dir / "errors.log"
    return output_dir, metadata_path, error_log_path

def main():
    parser = argparse.ArgumentParser(description="감정 음성 전처리 파이프라인")
    parser.add_argument("-i", "--input_dir", type=str, help="원본 오디오 폴더")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="출력 데이터셋 폴더")
    parser.add_argument("-f", "--one_file", type=str, help="파일 하나 지정")
    parser.add_argument("--whisper_model", type=str, default="large-v3")
    args = parser.parse_args()

    # 설정
    config = Config(
        whisper_model_size=args.whisper_model,
    )

    # 파일 하나만 test
    if args.one_file:
        o_path, *_ = make_dir(args.output_dir)
        # 처리
        with ThreadPoolExecutor(max_workers=2) as executor:
            preprocessor = Preprocessor(config, executor)
            test_result, _ = preprocessor.process_file(
                audio_path=args.one_file,
                output_dir=o_path,
                start_id=0,
            )
            print(test_result)
    else:
        o_path, m_path, e_path = make_dir(arg.output_dir)

        # 이어하기 상태 복원
        processed_sources, next_id = load_processed_state(m_path)
        if processed_sources:
            print(f"[이어하기] 기존 처리 {len(processed_sources)}개 파일, 다음 chunk_id={next_id}")

        # 오디오 파일 수집
        audio_files: list[Path] = []
        for ext in config.extensions:
            audio_files.extend(Path(args.input_dir).glob(f"*.{ext}"))
        audio_files = sorted(set(audio_files))

        # 이미 처리한 파일 스킵
        remains = [p for p in audio_files if p.stem not in processed_sources]
        print(f"총 {len(audio_files)}개 / 처리 대상 {len(remains)}개\n")

        # 처리
        with ThreadPoolExecutor(max_workers=2) as executor, \
            open(m_path, "a", encoding="utf-8") as f_meta, \
            open(e_path, "a", encoding="utf-8") as f_err, \
            tqdm(desc="처리 중", unit="file", total=len(remains)) as pbar:
                
            preprocessor = Preprocessor(config, executor)
            total_written = preprocessor.process_pipeline(
                audio_files=remains,
                output_dir=o_path,
                start_id=next_id,
                f_meta=f_meta,
                f_err=f_err,
                pbar=pbar,
            )

        print("\n" + "@" * 60)
        print(f"발화: {total_written}")
        print(f"메타데이터: {m_path}")
        print(f"청크 오디오: {o_path / 'chunks'}")
        print(f"감정 임베딩: {o_path / 'embeddings'}")
        print("@" * 60)

if __name__ == "__main__":
    main()
