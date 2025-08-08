# Tokenizer.py 详细实现文档

## 文件概述
- **路径**: `/tokenizer.py`
- **作用**: 音频到离散token的双码本编码器
- **核心技术**: FunASR + Whisper + ONNX，实现语言级和声学级的并行编码
- **依赖**: torch, torchaudio, onnxruntime, whisper, funasr_detach

## 类定义: StepAudioTokenizer

### 1. 初始化方法 `__init__`
```python
def __init__(self, encoder_path)
```

**初始化组件**:

1. **FunASR模型** (语言级编码):
   ```python
   funasr_model_path = "dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
   self.funasr_model = AutoModel(model=funasr_model_path)
   ```
   - Paraformer大模型
   - 支持中文、粤语、英文
   - 16kHz采样率

2. **K-means聚类中心**:
   ```python
   self.kms = torch.tensor(np.load(kms_path))
   ```
   - 用于量化语言特征
   - 1024个聚类中心

3. **ONNX推理会话** (声学级编码):
   ```python
   self.ort_session = onnxruntime.InferenceSession(
       cosy_tokenizer_path,
       sess_options=session_option,
       providers=["CUDAExecutionProvider"]
   )
   ```
   - GPU加速推理
   - 优化图级别: ORT_ENABLE_ALL

4. **流式处理参数**:
   ```python
   self.chunk_size = [0, 4, 5]
   self.encoder_chunk_look_back = 4
   self.decoder_chunk_look_back = 1
   ```

5. **会话管理**:
   ```python
   self.vq02_sessions = {}  # 流式会话缓存
   self.vq02_lock = threading.Lock()
   self.vq06_lock = threading.Lock()
   ```

### 2. 主调用方法 `__call__`
```python
def __call__(self, audio, sr)
```

**功能**: 音频到token字符串的完整转换

**流程**:
1. 调用`wav2token()`获取双码本
2. 调用`merge_vq0206_to_token_str()`合并
3. 返回格式化的token字符串

### 3. 音频预处理 `preprocess_wav`
```python
def preprocess_wav(self, audio, sample_rate, enable_trim=True, energy_norm=True)
```

**处理步骤**:
1. **重采样到16kHz**:
   ```python
   audio = resample_audio(audio, sample_rate, 16000)
   ```

2. **能量归一化** (可选):
   ```python
   if energy_norm:
       audio = energy_norm_fn(audio)
   ```

3. **静音裁剪** (可选):
   ```python
   if enable_trim:
       audio = trim_silence(audio, 16000)
   ```

### 4. 双码本编码 `wav2token`
```python
def wav2token(self, audio, sample_rate, enable_trim=True, energy_norm=True)
```

**核心流程**:

1. **预处理音频**:
   ```python
   audio = self.preprocess_wav(audio, sample_rate, enable_trim, energy_norm)
   ```

2. **语言级编码** (vq02):
   ```python
   vq02_ori = self.get_vq02_code(audio)
   vq02 = [int(x) + 65536 for x in vq02_ori]  # 偏移到音频token范围
   ```
   - 码率: 16.7Hz
   - 码本: 1024维

3. **声学级编码** (vq06):
   ```python
   vq06_ori = self.get_vq06_code(audio)
   vq06 = [int(x) + 65536 + 1024 for x in vq06_ori]  # 偏移避免冲突
   ```
   - 码率: 25Hz
   - 码本: 4096维

4. **2:3交错合并**:
   ```python
   chunk = 1
   chunk_nums = min(len(vq06) // (3 * chunk), len(vq02) // (2 * chunk))
   speech_tokens = []
   for idx in range(chunk_nums):
       speech_tokens += vq02[idx * chunk * 2 : (idx + 1) * chunk * 2]
       speech_tokens += vq06[idx * chunk * 3 : (idx + 1) * chunk * 3]
   ```
   - 每2个vq02对应3个vq06
   - 保持时序对齐

### 5. 语言级编码 `get_vq02_code`
```python
def get_vq02_code(self, audio, session_id=None, is_final=True)
```

**特点**:
- 支持流式处理（通过session_id）
- 缓存机制提高效率

**处理流程**:

1. **准备音频**:
   ```python
   _tmp_wav = io.BytesIO()
   torchaudio.save(_tmp_wav, audio, 16000, format="wav")
   ```

2. **编码器推理**:
   ```python
   res, new_cache = self.funasr_model.infer_encoder(
       input=[_tmp_wav],
       chunk_size=self.chunk_size,
       encoder_chunk_look_back=self.encoder_chunk_look_back,
       decoder_chunk_look_back=self.decoder_chunk_look_back,
       device=0,
       is_final=is_final,
       cache=cache
   )
   ```

3. **K-means量化**:
   ```python
   feat = res_["enc_out"]
   c_list = self.dump_label([feat], self.kms)[0]
   ```

4. **会话管理**:
   - 流式: 保存缓存
   - 最终: 清理缓存

### 6. 声学级编码 `get_vq06_code`
```python
def get_vq06_code(self, audio)
```

**特点**:
- 30秒分块处理防止内存溢出
- 使用Whisper提取mel频谱

**处理流程**:

1. **音频分块** (30秒):
   ```python
   def split_audio(audio, chunk_duration=480000):  # 30s * 16000
       chunks = []
       while start < len(audio):
           end = min(start + chunk_duration, len(audio))
           chunks.append(audio[start:end])
           start = end
   ```

2. **特征提取**:
   ```python
   feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
   ```

3. **ONNX推理**:
   ```python
   chunk_token = self.ort_session.run(
       None,
       {
           self.ort_session.get_inputs()[0].name: feat.numpy(),
           self.ort_session.get_inputs()[1].name: feat_len
       }
   )[0].flatten().tolist()
   ```

4. **验证码率**:
   ```python
   assert abs(len(chunk_token) - duration * 25) <= 2
   ```

### 7. K-means聚类 `kmean_cluster`
```python
def kmean_cluster(self, samples, means)
```

**功能**: 将连续特征量化为离散索引

**实现**:
```python
dists = torch.cdist(samples, means)  # 计算距离
indices = dists.argmin(dim=1).cpu().numpy()  # 最近邻
return indices.tolist()
```

### 8. 标签生成 `dump_label`
```python
def dump_label(self, samples, mean)
```

**功能**: 批量处理多个样本的量化

**步骤**:
1. 拼接所有样本
2. 执行K-means聚类
3. 按原始长度分割结果

### 9. 双码本合并 `merge_vq0206_to_token_str`
```python
def merge_vq0206_to_token_str(self, vq02, vq06)
```

**功能**: 将两个码本按2:3比例交错合并

**实现**:
```python
_vq06 = [1024 + x for x in vq06]  # 偏移避免冲突
result = []
i = 0
j = 0
while i < len(vq02) - 1 and j < len(_vq06) - 2:
    sublist = vq02[i : i + 2] + _vq06[j : j + 3]
    result.extend(sublist)
    i += 2
    j += 3
return "".join([f"<audio_{x}>" for x in result])
```

## 关键技术细节

### 1. 双码本设计理由
- **语言级(vq02)**: 捕获语义信息，低码率(16.7Hz)
- **声学级(vq06)**: 捕获韵律细节，高码率(25Hz)
- **优势**: 分离语义和韵律，提高压缩效率

### 2. Token ID分配
```
文本token: 0-65535
vq02 token: 65536-66559 (65536 + 0~1023)
vq06 token: 66560-70655 (65536 + 1024 + 0~4095)
```

### 3. 时序对齐
- vq02: 16.7Hz → 60ms/token
- vq06: 25Hz → 40ms/token
- 最小公倍数: 120ms
- 对齐方案: 2个vq02(120ms) = 3个vq06(120ms)

### 4. 流式处理优化
- 会话缓存机制
- 增量编码支持
- 线程安全设计

## 性能分析

### 1. 计算瓶颈
- FunASR编码器: ~100ms/秒音频
- ONNX推理: ~50ms/秒音频
- K-means量化: ~10ms/秒音频

### 2. 内存占用
- FunASR模型: ~1GB
- ONNX模型: ~500MB
- K-means中心: ~4MB

### 3. 优化建议
1. 批处理多个音频
2. 异步并行处理vq02和vq06
3. 使用更小的量化模型
4. 实现真正的流式管道

## 使用示例

### 基本编码
```python
tokenizer = StepAudioTokenizer(encoder_path)
audio, sr = torchaudio.load("audio.wav")
tokens = tokenizer(audio, sr)
# 输出: "<audio_65536><audio_65537>..."
```

### 流式编码
```python
session_id = "user_123"
# 第一块
tokens1 = tokenizer.get_vq02_code(audio1, session_id, is_final=False)
# 第二块
tokens2 = tokenizer.get_vq02_code(audio2, session_id, is_final=True)
```

## 错误处理建议

1. 音频长度检查（最小480采样点）
2. 采样率验证
3. ONNX会话异常处理
4. 内存溢出保护

## 扩展方向

1. **多码本支持**: 扩展到3个或更多码本
2. **自适应码率**: 根据内容复杂度调整
3. **端到端训练**: 联合优化双码本
4. **压缩优化**: 进一步降低token率