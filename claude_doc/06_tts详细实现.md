# TTS.py 详细实现文档

## 文件概述
- **路径**: `/tts.py`
- **作用**: 文本到语音的生成引擎，支持情感控制、方言、音乐和语音克隆
- **核心模型**: 3B参数LLM + CosyVoice声码器
- **依赖**: transformers, torch, cosyvoice

## 类定义: StepAudioTTS

### 1. 初始化方法 `__init__`
```python
def __init__(self, model_path, encoder)
```

**初始化组件**:

1. **加载自定义库**:
   ```python
   load_optimus_ths_lib(os.path.join(model_path, 'lib'))
   ```

2. **3B TTS语言模型**:
   ```python
   self.llm = AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype=torch.bfloat16,
       device_map="cuda",
       trust_remote_code=True
   )
   ```

3. **双声码器配置**:
   ```python
   # 常规语音
   self.common_cosy_model = CosyVoice(
       os.path.join(model_path, "CosyVoice-300M-25Hz")
   )
   # 音乐语音(RAP/哼唱)
   self.music_cosy_model = CosyVoice(
       os.path.join(model_path, "CosyVoice-300M-25Hz-Music")
   )
   ```

4. **系统提示词**:
   ```python
   self.sys_prompt_dict = {
       "sys_prompt_for_rap": "请参考对话历史里的音色，用RAP方式...",
       "sys_prompt_for_vocal": "请参考对话历史里的音色，用哼唱的方式...",
       "sys_prompt_wo_spk": "作为一名卓越的声优演员...",
       "sys_prompt_with_spk": "作为一名卓越的声优演员...使用[{}]的声音..."
   }
   ```

5. **注册预设音色**:
   ```python
   self.register_speakers()
   ```

### 2. 自定义LogitsProcessor
```python
class RepetitionAwareLogitsProcessor(LogitsProcessor)
```

**功能**: 防止token重复，提高生成质量

**实现**:
```python
def __call__(self, input_ids, scores):
    window_size = 10
    threshold = 0.1
    
    # 检查最近10个token的重复率
    window = input_ids[:, -window_size:]
    last_tokens = window[:, -1].unsqueeze(-1)
    repeat_counts = (window == last_tokens).sum(dim=1)
    repeat_ratios = repeat_counts.float() / window_size
    
    # 如果重复率超过阈值，禁止该token
    mask = repeat_ratios > threshold
    scores[mask, last_tokens[mask].squeeze(-1)] = float("-inf")
    return scores
```

### 3. 主调用方法 `__call__`
```python
def __call__(self, text: str, prompt_speaker: str, clone_dict: dict | None = None)
```

**参数**:
- `text`: 要合成的文本
- `prompt_speaker`: 预设音色ID
- `clone_dict`: 克隆信息字典(可选)

**克隆字典格式**:
```python
{
    "speaker": "speaker_id",
    "prompt_text": "提示文本内容",
    "wav_path": "提示音频路径"
}
```

**处理流程**:

1. **处理克隆音频** (如果提供):
   ```python
   if clone_dict:
       clone_prompt_code, clone_prompt_token, ... = 
           self.preprocess_prompt_wav(clone_dict['wav_path'])
       # 构建克隆音色信息
       clone_speakers_info = {
           "prompt_text": clone_dict['prompt_text'],
           "prompt_code": clone_prompt_code,
           "cosy_speech_feat": clone_speech_feat,
           ...
       }
   ```

2. **检测指令类型**:
   ```python
   instruction_name = self.detect_instruction_name(text)
   ```
   - 识别RAP、哼唱等特殊指令
   - 根据指令选择合适的声码器

3. **选择声码器**:
   ```python
   if instruction_name in ("RAP", "哼唱"):
       cosy_model = self.music_cosy_model
       # 使用音乐版音色
       if not clone_dict:
           prompt_speaker_info = self.speakers_info[f"{prompt_speaker}{instruction_name}"]
   else:
       cosy_model = self.common_cosy_model
   ```

4. **构建输入序列**:
   ```python
   token_ids = self.tokenize(
       text,
       prompt_speaker_info["prompt_text"],
       prompt_speaker,
       prompt_speaker_info["prompt_code"]
   )
   ```

5. **生成音频token**:
   ```python
   output_ids = self.llm.generate(
       torch.tensor([token_ids]).to("cuda"),
       max_length=8192,
       temperature=0.7,
       do_sample=True,
       logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()])
   )
   ```

6. **token转音频**:
   ```python
   return cosy_model.token_to_wav_offline(
       output_ids - 65536,  # 还原到原始范围
       prompt_speaker_info["cosy_speech_feat"],
       prompt_speaker_info["cosy_speech_feat_len"],
       prompt_speaker_info["cosy_prompt_token"],
       prompt_speaker_info["cosy_prompt_token_len"],
       prompt_speaker_info["cosy_speech_embedding"]
   ), 22050  # 返回音频和采样率
   ```

### 4. 音色注册 `register_speakers`
```python
def register_speakers(self)
```

**功能**: 加载预设音色库

**处理流程**:
1. 读取speakers_info.json
2. 对每个音色:
   - 加载提示音频
   - 提取特征
   - 保存到self.speakers_info

**音色类型**:
- 常规音色: Tingting
- RAP音色: TingtingRAP
- 哼唱音色: Tingting哼唱

### 5. 指令检测 `detect_instruction_name`
```python
def detect_instruction_name(self, text)
```

**功能**: 从文本开头的括号中提取控制指令

**实现**:
```python
match_group = re.match(r"^([（\(][^\(\)()]*[）\)]).*$", text, re.DOTALL)
if match_group is not None:
    instruction = match_group.group(1)
    instruction_name = instruction.strip("()（）")
return instruction_name
```

**支持的指令**:
- 情感: (高兴1)、(生气1)、(悲伤1)、(撒娇1)
- 语言: (英文)、(日语)、(韩语)
- 方言: (粤语)、(四川话)
- 音乐: (RAP)、(哼唱)
- 语速: (慢速1)、(快速1)

### 6. 输入序列构建 `tokenize`
```python
def tokenize(self, text: str, prompt_text: str, 
            prompt_speaker: str, prompt_code: list)
```

**功能**: 构建TTS模型的输入token序列

**序列结构**:
```
[BOS] [BOT] system
{系统提示词}
[EOT] [BOT] human
{提示文本}
[EOT] [BOT] assistant
{提示音频token}
[EOT] [BOT] human
{目标文本}
[EOT] [BOT] assistant
```

**实现细节**:

1. **选择系统提示**:
   ```python
   if rap_or_vocal:
       if "哼唱" in text:
           prompt = self.sys_prompt_dict["sys_prompt_for_vocal"]
       else:
           prompt = self.sys_prompt_dict["sys_prompt_for_rap"]
   elif prompt_speaker:
       prompt = self.sys_prompt_dict["sys_prompt_with_spk"].format(prompt_speaker)
   else:
       prompt = self.sys_prompt_dict["sys_prompt_wo_spk"]
   ```

2. **构建token序列**:
   ```python
   history = [1]  # BOS
   history.extend([4] + sys_tokens + [3])  # system
   history.extend([4] + qrole_toks + prompt_tokens + [3])  # prompt text
   history.extend([4] + arole_toks + prompt_code + [3])  # prompt audio
   history.extend([4] + qrole_toks + target_tokens + [3])  # target text
   history.extend([4] + arole_toks)  # assistant开始
   ```

### 7. 提示音频处理 `preprocess_prompt_wav`
```python
def preprocess_prompt_wav(self, prompt_wav_path: str)
```

**功能**: 处理克隆或提示音频，提取必要特征

**处理步骤**:

1. **加载音频**:
   ```python
   prompt_wav, prompt_wav_sr = torchaudio.load(prompt_wav_path)
   if prompt_wav.shape[0] > 1:
       prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # 转单声道
   ```

2. **多采样率处理**:
   ```python
   # ASR需要16kHz
   prompt_wav_16k = torchaudio.transforms.Resample(
       orig_freq=prompt_wav_sr, new_freq=16000
   )(prompt_wav)
   
   # TTS需要22.05kHz
   prompt_wav_22k = torchaudio.transforms.Resample(
       orig_freq=prompt_wav_sr, new_freq=22050
   )(prompt_wav)
   ```

3. **提取CosyVoice特征**:
   ```python
   # 语音特征(用于Flow Matching)
   speech_feat, speech_feat_len = 
       self.common_cosy_model.frontend._extract_speech_feat(prompt_wav_22k)
   
   # 说话人嵌入(用于音色控制)
   speech_embedding = 
       self.common_cosy_model.frontend._extract_spk_embedding(prompt_wav_16k)
   ```

4. **编码音频token**:
   ```python
   prompt_code, _, _ = self.encoder.wav2token(prompt_wav, prompt_wav_sr)
   prompt_token = torch.tensor([prompt_code], dtype=torch.long) - 65536
   prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long)
   ```

**返回值**:
- prompt_code: 音频token列表
- prompt_token: tensor格式token
- prompt_token_len: token长度
- speech_feat: 语音特征
- speech_feat_len: 特征长度
- speech_embedding: 说话人嵌入

## 关键技术细节

### 1. 双声码器设计
- **常规版**: 适用于普通语音，优化自然度
- **音乐版**: 适用于RAP/唱歌，优化韵律表现

### 2. Token ID映射
```python
# 编码时: 原始ID + 65536
encoded_token = original_token + 65536

# 解码时: 减去65536还原
original_token = encoded_token - 65536
```

### 3. 情感控制机制
- 通过括号标签实现
- 系统提示词引导模型理解
- 不同情感使用不同的生成策略

### 4. 克隆流程
```
原始音频 → 特征提取 → 说话人嵌入
                    ↓
目标文本 → LLM生成 → 个性化token
                    ↓
            CosyVoice解码 → 克隆音频
```

## 使用示例

### 基础TTS
```python
tts = StepAudioTTS(model_path, encoder)
audio, sr = tts("你好世界", "Tingting")
```

### 情感控制
```python
audio, sr = tts("(高兴1)今天天气真好!", "Tingting")
```

### RAP生成
```python
audio, sr = tts("(RAP)我是最强的说唱歌手", "Tingting")
```

### 语音克隆
```python
clone_dict = {
    "speaker": "custom_voice",
    "prompt_text": "这是一段示例文本",
    "wav_path": "path/to/reference.wav"
}
audio, sr = tts("克隆后的语音", "", clone_dict)
```

## 性能优化

### 1. 推理优化
- 使用bfloat16减少内存
- RepetitionAwareLogitsProcessor提高质量
- 批处理支持(待实现)

### 2. 缓存机制
- 预加载所有预设音色
- 特征提取结果缓存

### 3. 并行处理
- 双声码器可并行加载
- 特征提取可异步进行

## 扩展方向

1. **更多控制维度**: 音调、音量、共鸣等
2. **流式合成**: 逐字生成减少延迟
3. **多说话人**: 单句内切换音色
4. **风格迁移**: 保留内容改变风格
5. **情感强度**: 支持连续情感控制