import torchaudio
import argparse
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from utils import load_audio
import os


def main():
    parser = argparse.ArgumentParser(description="StepAudio Offline Inference")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Base path for model files"
    )
    parser.add_argument(
        "--synthesis-type", type=str, default="tts", help="Use tts or Clone for Synthesis"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output path for synthesis audios"
    )
    args = parser.parse_args()
    os.makedirs(f"{args.output_path}", exist_ok=True)

    encoder = StepAudioTokenizer(f"{args.model_path}/Step-Audio-Tokenizer")
    tts_engine = StepAudioTTS(f"{args.model_path}/Step-Audio-TTS-3B", encoder)

    text = "（RAP）我踏上自由的征途，追逐那遥远的梦想，挣脱束缚的枷锁，让心灵随风飘荡，每一步都充满力量，每一刻都无比闪亮，自由的信念在燃烧，照亮我前进的方向!"
    #output_audio, sr = tts_engine("（RAP）我想要问你究竟想要嫁给谁，我不是你的王子，我的摩托是哈雷，如果你想要爱我，就填饱我的胃，别看我身上装扮，它一定比你贵。", "闫雨婷")
    output_audio, sr = tts_engine(text, "闫雨婷")
    torchaudio.save(f"{args.output_path}/output_tts.wav", output_audio, sr)

    text_clone = "我一边走一边看，看看人来的人往的"
    #clone_speaker = {'speaker':'test','prompt_text':'那等我们到海洋馆之后，给妈妈买个礼物，好不好呀？', 'wav_path':'speakers/闫雨婷_prompt.wav'}
    clone_speaker = {'speaker':'test','prompt_text':'本人虽说村长落选，但思想工作还是要搞，在家开个心理诊所', 'wav_path':'examples/prompt_wav_zhaobenshan.wav'}
    #output_audio, sr = tts_engine("今天天气不错", "",clone_speaker)
    output_audio, sr = tts_engine(text_clone, "",clone_speaker)
    torchaudio.save(f"{args.output_path}/output_clone.wav", output_audio, sr)

if __name__ == "__main__":
    main()
