import gradio as gr
import time
from pathlib import Path
import torchaudio
from stepaudio import StepAudio

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

CACHE_DIR = "/tmp/gradio/"
system_promtp = {"role": "system", "content": "适配用户的语言，用口语化的文字回答"}


class CustomAsr:
    def __init__(self, model_name="iic/SenseVoiceSmall", device="cuda"):
        self.model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
        )

    def run(self, audio_path):
        res = self.model.generate(
            input=audio_path,
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text


def add_message(chatbot, history, mic, text, asr_model):
    if not mic and not text:
        return chatbot, history, "Input is empty"

    if text:
        chatbot.append({"role": "user", "content": text})
        history.append({"role": "user", "content": text})
    elif mic and Path(mic).exists():
        chatbot.append({"role": "user", "content": {"path": mic}})
        # 此处直接使用用户语音的 asr 结果进行推理
        text = asr_model.run(mic)
        chatbot.append({"role": "user", "content": text})
        history.append({"role": "user", "content": text})

    print(f"{chatbot=}")
    print(f"{history=}")
    return chatbot, history, None


def reset_state():
    """Reset the chat history."""
    return [], [system_promtp]


def save_tmp_audio(audio, sr):
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=CACHE_DIR, delete=False, suffix=".wav"
    ) as temp_audio:
        temp_audio_path = temp_audio.name
        torchaudio.save(temp_audio_path, audio, sr)

    return temp_audio.name


# 将 history 给模型进行推理,结果保存 history 和 chatbot
def predict(chatbot, history, audio_model):
    """Generate a response from the model."""
    try:
        text, audio, sr = audio_model(history, "闫雨婷")
        print(f"predict {text=}")
        audio_path = save_tmp_audio(audio, sr)
        chatbot.append({"role": "assistant", "content": {"path": audio_path}})
        chatbot.append({"role": "assistant", "content": text})
        history.append({"role": "assistant", "content": text})
    except Exception as e:
        print(e)
        gr.Warning(f"Some error happend, retry submit")
    return chatbot, history


def _launch_demo(args, audio_model, asr_model):
    with gr.Blocks(delete_cache=(86400, 86400)) as demo:
        gr.Markdown("""<center><font size=8>Step1o Audio Chat</center>""")
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            avatar_images=["assets/user.png", "assets/assistant.png"],
            min_height=800,
            type="messages",
        )
        # 保存 chat 历史，不需要每次再重新拼格式
        history = gr.State([system_promtp])
        mic = gr.Audio(type="filepath")
        text = gr.Textbox(placeholder="Enter message ...")

        with gr.Row():
            clean_btn = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            submit_btn = gr.Button("🚀 Submit")

        def on_submit(chatbot, history, mic, text):
            chatbot, history, error = add_message(
                chatbot, history, mic, text, asr_model
            )
            if error:
                gr.Warning(error)  # 显示警告消息
                return chatbot, history, None, None
            else:
                chatbot, history = predict(chatbot, history, audio_model)
                return chatbot, history, None, None

        submit_btn.click(
            fn=on_submit,
            inputs=[chatbot, history, mic, text],
            outputs=[chatbot, history, mic, text],
            concurrency_limit=4,
            concurrency_id="gpu_queue",
        )
        clean_btn.click(
            reset_state,
            outputs=[chatbot, history],
            show_progress=True,
        )

        def regenerate(chatbot, history):
            while chatbot and chatbot[-1]["role"] == "assistant":
                print(f"discard {chatbot[-1]}")
                chatbot.pop()
            while history and history[-1]["role"] == "assistant":
                print(f"discard {history[-1]}")
                history.pop()
            return predict(chatbot, history, audio_model)

        regen_btn.click(
            regenerate,
            [chatbot, history],
            [chatbot, history],
            show_progress=True,
            concurrency_id="gpu_queue",
        )

    demo.queue().launch(
        share=False,
        server_port=args.server_port,
        server_name=args.server_name,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Model path.")
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )
    args = parser.parse_args()

    audio_model = StepAudio(
        tokenizer_path=os.path.join(args.model_path, "step-audio-tokenizer"),
        tts_path=os.path.join(args.model_path, "step-audio-tts-3b"),
        chat_path=os.path.join(args.model_path, "step-audio-chat"),
    )
    asr_model = CustomAsr()
    _launch_demo(args, audio_model, asr_model)
