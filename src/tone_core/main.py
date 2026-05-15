import numpy as np

from .config import SenderConfig, ReceiverConfig
from .sender import SenderEncode
from .receiver import ReceiverDecode

# 예시
# mac에서 할 때 config.py 에서 device 설정 바꾸기
'''
    encode()
    
    Input:
    audio: np.ndarray[np.float32]
    
    Return:
    EncodeResult(text='', emotion_indices=(4, 4, 4, 6, 7, 1, 7, 0), emotion_label=<EmotionLabel.NEUTRAL: 4>, emotion_label_idx=4, emotion_score=0.8566851019859314)

'''

model_path = "/home/cap/data/models/skip_kl_8d_8L_kl05_1e-4.pt"

def main():
    sample_rate = 16000
    duration = 1.0
    arr = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    cfgS = SenderConfig(fsq_ckpt = model_path)
    sender = SenderEncode.from_config(cfgS)
    cfgR = ReceiverConfig(fsq_ckpt = model_path)
    receiver = ReceiverDecode.from_config(cfgR)
    
    result = sender.encode(arr)
    # print(result)
    
    emo_vec_1024d = receiver.decode(result.emotion_indices)
    # print(emo_vec_1024d)

if __name__ == "__main__":
    main()