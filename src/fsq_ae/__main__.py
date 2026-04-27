import argparse
 
from .config import FSQAEConfig
from .prepare_data import prepare
from .train import train
 
 
def main():
    parser = argparse.ArgumentParser("fsq_ae")
    sub = parser.add_subparsers(dest="cmd", required=True)
 
    # prepare-data: 외부 입력 경로 + 출력 경로만 받음
    p1 = sub.add_parser("prepare-data",
                        help="JSONL + .npy 파일들을 단일 .pt로 합치기")
    p1.add_argument("--metadata", type=str, required=True)
    p1.add_argument("--emb_dir", type=str, required=True)
    p1.add_argument("--out_path", type=str, default=None,
                    help="미지정 시 config.data_path 사용")
 
    sub.add_parser("train", help="config.py 설정대로 학습")
 
    args = parser.parse_args()
 
    if args.cmd == "prepare-data":
        out_path = args.out_path or FSQAEConfig().data_path
        prepare(args.metadata, args.emb_dir, out_path)
 
    elif args.cmd == "train":
        train(FSQAEConfig())
 
 
if __name__ == "__main__":
    main()