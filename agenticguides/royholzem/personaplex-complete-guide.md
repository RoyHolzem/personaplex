# PersonaPlex: Complete System Guide for AI Agents

## Executive Summary

PersonaPlex is a production-ready, full-duplex speech-to-speech conversational AI system developed by NVIDIA. It enables real-time spoken interactions with controllable persona (via text prompts) and voice characteristics (via audio embeddings). The system is built on the Moshi architecture and consists of three primary components: the Mimi audio codec, a 7B parameter language model, and a streaming generation engine.

**Key Capabilities:**
- Real-time full-duplex conversation (both parties can speak simultaneously)
- Persona control through text-based role descriptions
- Voice control through pre-computed audio embeddings
- Low-latency processing (~80ms per frame at 12.5Hz)
- Natural handling of interruptions, backchanneling, and turn-taking

## System Architecture

### High-Level Overview

```
Input Audio (24kHz) 
    ↓
[Mimi Encoder] → Audio Codes (8 codebooks)
    ↓
[LMModel: Main Transformer + Depformer]
    ↓ (generates text + 8 audio codebooks)
[Mimi Decoder] → Output Audio (24kHz)
    ↓
Output Audio + Text Transcription
```

### Component 1: Mimi Audio Codec

**Purpose**: Compress raw audio into discrete token sequences for the language model

**Architecture**:
- **Encoder**: SEANetEncoder with 512-dimensional latent space
- **Decoder**: SEANetDecoder (symmetric to encoder)
- **Quantizer**: SplitResidualVectorQuantizer with 32 codebooks (uses 8 for inference)
- **Transformers**: Optional encoder/decoder transformers for improved quality

**Parameters**:
- Sample rate: 24,000 Hz
- Frame rate: 12.5 Hz (one frame = 1,920 samples = 80ms)
- Compression ratios: [8, 6, 5, 4] (total 960x compression)
- Codebook cardinality: 2,048 tokens per codebook
- Active codebooks: 8 (out of 32 available)

**Key Methods**:
- `encode(audio_tensor)`: [B, 1, T] → [B, 8, F] where F = T / 1920
- `decode(codes)`: [B, 8, F] → [B, 1, T]
- `set_num_codebooks(n)`: Configure active codebook count

**File Location**: `moshi/moshi/models/compression.py`

### Component 2: LMModel (Language Model)

**Purpose**: Joint modeling of text and audio token sequences

**Architecture**: Dual-transformer system
1. **Main Transformer**:
   - Dimensions: 4096
   - Layers: 32
   - Heads: 32
   - Context: 3000 tokens
   - Processes all 17 input streams (1 text + 16 audio)

2. **Depformer (Depth Transformer)**:
   - Dimensions: 1024
   - Layers: 6
   - Heads: 16
   - Context: 8 steps
   - Processes inter-codebook dependencies for the 8 output audio streams

**Token Streams**:
- Stream 0: Text tokens (vocabulary: 32,000 from SentencePiece)
- Streams 1-8: Moshi output audio (8 codebooks)
- Streams 9-16: User input audio (8 codebooks)

**Delay Pattern**: `[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`
- Enables causal generation while maintaining real-time performance
- Delays compensate for autoregressive dependencies

**Special Tokens**:
- `initial_token_id`: 2048 (start of audio sequence)
- `text_initial_token_id`: 32000 (start of text sequence)
- `text_padding_token_id`: 3 or 32000 (depending on tokenizer)
- `zero_token_id`: -1 (no sampling/input for this position)
- `ungenerated_token_id`: -2 (to be predicted/sampled)

**Forward Pass**:
```python
# Input: [B, 17, T] token tensor
# Output: LMOutput with:
#   - logits: [B, 8, T, 2048] for audio
#   - text_logits: [B, 1, T, 32000] for text
#   - masks: indicating valid positions
```

**File Location**: `moshi/moshi/models/lm.py` (class `LMModel`)

### Component 3: LMGen (Generation Engine)

**Purpose**: Manage streaming inference, prompting, and token sampling

**Key Responsibilities**:
1. Initialize and maintain streaming state
2. Load and process text prompts (system messages)
3. Load and process voice prompts (audio embeddings)
4. Perform autoregressive generation with temperature/top-k sampling
5. Manage token caching and delay compensation

**Prompting System**:

**Phase 1: Text Prompt**
- System message wrapped in `<system>` tags
- Encoded with SentencePiece tokenizer
- Fed into model during warmup phase
- Defines persona, role, and context

**Phase 2: Voice Prompt**
- 3-second audio sample (typically)
- Encoded through Mimi to get audio codes
- Processed to extract voice embeddings
- Can be:
  - Live audio file (`.wav`)
  - Pre-computed embeddings (`.pt` file)

**Phase 3: Silence Padding**
- Generates silence/sine wave frames
- Allows model to "settle" into the persona
- Typically 0.5 seconds (6 frames at 12.5Hz)

**Sampling Parameters**:
- `temp`: Temperature for audio tokens (default: 0.8)
- `temp_text`: Temperature for text tokens (default: 0.7)
- `top_k`: Top-k filtering for audio (default: 250)
- `top_k_text`: Top-k filtering for text (default: 25)

**CUDA Optimization**:
- Uses CUDA graphs for zero-overhead kernel launches
- Three graphed functions:
  - `forward_codes`: Main model forward pass
  - `forward_embeddings`: Embedding computation
  - `depformer_step`: Depformer computation

**State Management**:
```python
class _LMGenState:
    cache: torch.Tensor        # [B, 17, max_delay+3] token cache
    provided: torch.Tensor     # [B, 17, max_delay+3] provided mask
    initial: torch.Tensor      # Initial tokens for each stream
    offset: int                # Current generation step
    graphed_*: CUDAGraphed     # CUDA graph wrappers
```

**File Location**: `moshi/moshi/models/lm.py` (class `LMGen`)

## Deployment Modes

### Mode 1: Server (Real-Time WebSocket)

**Entry Point**: `python -m moshi.server`

**Architecture**:
```
Client (Browser)
    ↓ WebSocket (HTTPS)
Server (Python aiohttp)
    ↓
[Opus Encoder/Decoder] ← sphn library
    ↓
[Mimi + LMModel + LMGen]
    ↓
WebSocket Messages:
  - 0x00: Handshake
  - 0x01: Audio data (Opus-encoded)
  - 0x02: Text transcription
```

**Key Features**:
- Async/await for concurrent request handling
- Lock-based serialization (one conversation at a time per instance)
- SSL/TLS support with automatic cert generation
- Query parameters for configuration:
  - `text_prompt`: System message
  - `voice_prompt`: Voice file name
  - `seed`: Random seed for reproducibility

**Connection Flow**:
1. Client connects via WebSocket
2. Server loads prompts and initializes state
3. Server runs system prompt phases
4. Server sends handshake byte (0x00)
5. Three concurrent loops:
   - `recv_loop()`: Receive Opus audio from client
   - `opus_loop()`: Process audio through model
   - `send_loop()`: Send generated audio back

**File Location**: `moshi/moshi/server.py`

### Mode 2: Offline (Batch Processing)

**Entry Point**: `python -m moshi.offline`

**Use Case**: Evaluation, testing, reproducible generation

**Process**:
1. Load input WAV file
2. Stream audio frames through model
3. Accumulate output frames
4. Write output WAV file (same duration as input)
5. Optionally write text transcript to JSON

**Example**:
```bash
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --text-prompt "You are a helpful assistant." \
  --input-wav "user_audio.wav" \
  --output-wav "assistant_audio.wav" \
  --output-text "transcript.json" \
  --seed 42424242
```

**File Location**: `moshi/moshi/offline.py`

### Mode 3: Docker

**Container**: CUDA-enabled Ubuntu 22.04 with PyTorch

**Configuration**:
```yaml
# docker-compose.yaml
services:
  personaplex:
    build: .
    ports:
      - "8998:8998"
    environment:
      - NO_TORCH_COMPILE=1
      - HF_TOKEN=${HF_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Usage**:
```bash
docker-compose up
```

## Web Client

### Technology Stack
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Routing**: React Router v6
- **Styling**: Tailwind CSS + DaisyUI
- **Audio**: Web Audio API + AudioWorklet

### Architecture

**Main Components**:

1. **Queue.tsx**: Landing page with configuration
   - Text prompt input (with presets)
   - Voice selection dropdown
   - Microphone permission handling
   - AudioContext initialization

2. **Conversation.tsx**: Active conversation interface
   - Real-time audio visualization
   - WebSocket connection management
   - Text transcription display
   - Disconnect/reconnect controls

3. **Audio Processing Pipeline**:
```
Microphone
    ↓
AudioWorkletNode (moshi-processor)
    ↓
Opus Encoding (opus-recorder library)
    ↓
WebSocket Send
    ↓
WebSocket Receive
    ↓
Opus Decoding (decoder worker)
    ↓
Audio Output
```

**WebSocket Protocol**:
- Message format: 1-byte type + payload
- Type 0x00: Handshake acknowledgment
- Type 0x01: Binary audio data (Opus frames)
- Type 0x02: Text transcription (UTF-8)

**File Location**: `client/src/`

## Voice Prompts

### Pre-packaged Voices

**Natural Voices** (more conversational):
- NATF0, NATF1, NATF2, NATF3 (female)
- NATM0, NATM1, NATM2, NATM3 (male)

**Variety Voices** (more diverse):
- VARF0, VARF1, VARF2, VARF3, VARF4 (female)
- VARM0, VARM1, VARM2, VARM3, VARM4 (male)

### Voice Prompt Format

**Audio Files** (`.wav`):
- Typically 3 seconds duration
- 24kHz sample rate
- Mono channel
- Processed through Mimi encoder to extract embeddings

**Pre-computed Embeddings** (`.pt`):
- PyTorch tensor saved with `torch.save()`
- Contains voice characteristics in latent space
- Faster loading (no encoding required)

### Creating Custom Voices

To create a custom voice prompt:

1. Record 3-5 seconds of clear speech
2. Convert to 24kHz mono WAV
3. Use existing server with voice prompt disabled
4. Or pre-compute embeddings:

```python
# Load audio
audio = load_audio("custom_voice.wav", 24000)

# Encode with Mimi
mimi = loaders.get_mimi("mimi_weights.safetensors", "cuda")
codes = mimi.encode(torch.tensor(audio).unsqueeze(0).unsqueeze(0))

# Extract embeddings from LMGen
lm_gen = LMGen(lm_model, device="cuda", save_voice_prompt_embeddings=True)
lm_gen.load_voice_prompt("custom_voice.wav")
embeddings = lm_gen.voice_prompt_embeddings

# Save for reuse
torch.save(embeddings, "custom_voice.pt")
```

## Text Prompts

### Prompt Structure

All text prompts are automatically wrapped in system tags:
```
<system> Your prompt text here <system>
```

### Prompt Categories

**1. Assistant Role**:
```
You are a wise and friendly teacher. Answer questions or provide 
advice in a clear and engaging way.
```
- General Q&A and information retrieval
- Used for FullDuplexBench "User Interruption" evaluation

**2. Customer Service Roles**:
```
You work for [Company] which is a [Type] and your name is [Name]. 
Information: [Specific details about products, policies, schedules, etc.]
```

Examples:
- Waste management: CitySan Services
- Restaurant: Jerusalem Shakshuka
- Drone rental: AeroRentals Pro
- Appliance repair: SwiftPlex Appliances

**3. Casual Conversations**:
```
You enjoy having a good conversation.
```
or with topic guidance:
```
You enjoy having a good conversation. Have a casual discussion 
about eating at home versus dining out.
```

Used for FullDuplexBench evaluations:
- Pause Handling
- Backchannel
- Smooth Turn Taking

**4. Creative/Out-of-Distribution**:
```
You enjoy having a good conversation. Have a technical discussion 
about fixing a reactor core on a spaceship to Mars. You are an 
astronaut on a Mars mission. Your name is Alex. You are already 
dealing with a reactor core meltdown...
```

These demonstrate emergent capabilities beyond training distribution.

### Best Practices

1. **Be Specific**: Include role, context, and constraints
2. **Include Identity**: Name and organization for service roles
3. **Provide Information**: Facts the assistant should know
4. **Set Boundaries**: What the assistant should/shouldn't do
5. **Keep Reasonable Length**: ~500-1000 characters optimal

## Training Data

### Synthetic Conversations

**Assistant Role**:
- Fixed prompt (teacher/advisor persona)
- Paired with diverse user questions
- Focus on clear, informative responses

**Customer Service Roles**:
- Variable prompts (different companies/scenarios)
- Structured information delivery
- Goal-oriented conversations

### Real Conversations

**Source**: Fisher English Corpus
- Telephone conversations in English
- Natural turn-taking and interruptions
- LLM-labeled with conversation topics and styles

**Prompts Derived**:
- Open-ended conversation prompts
- Topic-specific guidance
- Personality/style descriptors

## Model Weights

### HuggingFace Repository
**Repo**: `nvidia/personaplex-7b-v1`

**Files**:
- `model.safetensors`: LMModel weights (~14GB)
- `tokenizer-e351c8d8-checkpoint125.safetensors`: Mimi weights (~300MB)
- `tokenizer_spm_32k_3.model`: SentencePiece tokenizer
- `voices.tgz`: Pre-packaged voice embeddings
- `dist.tgz`: Web client static files
- `config.json`: Model configuration

### Loading Process

**Mimi**:
```python
from moshi.models import loaders

mimi = loaders.get_mimi(
    filename="path/to/mimi.safetensors",
    device="cuda"
)
```

**LMModel**:
```python
lm_model = loaders.get_moshi_lm(
    filename="path/to/model.safetensors",
    device="cuda",
    dtype=torch.bfloat16,
    cpu_offload=False  # Set True for CPU offload
)
```

**CPU Offload** (requires `accelerate`):
- Automatically distributes layers between GPU and CPU
- Uses `infer_auto_device_map` to determine placement
- Recommended for GPUs with <16GB VRAM

## Performance Optimization

### CUDA Graphs

**Purpose**: Eliminate kernel launch overhead for repeated operations

**Implementation**:
```python
from moshi.utils.compile import CUDAGraphed

# Wrap function
graphed_fn = CUDAGraphed(model.forward, disable=False)

# First call: records graph
output = graphed_fn(input_tensor)

# Subsequent calls: replays graph (much faster)
output = graphed_fn(input_tensor)
```

**Graphed Operations**:
1. `LMModel.forward_codes`: Main forward pass
2. `LMModel.forward_embeddings`: Embedding lookup
3. `LMGen.depformer_step`: Depformer processing

**Requirements**:
- CUDA device (automatically disabled on CPU)
- Fixed input/output shapes
- No dynamic control flow

### Streaming State Management

**Purpose**: Maintain model state across time steps without recomputation

**Key Concept**: Each module maintains internal state (KV cache, buffers, etc.)

**Interface**:
```python
# Initialize streaming
model.streaming_forever(batch_size=1)

# Reset between conversations
model.reset_streaming()

# Process frame-by-frame
for frame in audio_frames:
    codes = mimi.encode(frame)  # Uses streaming state
    tokens = lm_gen.step(codes)  # Uses streaming state
    audio = mimi.decode(tokens)  # Uses streaming state
```

**State Components**:
- Transformer KV caches
- Convolutional layer buffers
- Quantizer state
- Token history cache

### Memory Management

**GPU Memory Usage** (approximate):
- Mimi: ~1GB
- LMModel (bfloat16): ~14GB
- KV Cache: ~2GB per conversation
- **Total**: ~17-20GB for active inference

**Optimization Strategies**:
1. **CPU Offload**: Move least-used layers to CPU
2. **Batch Size**: Always use batch_size=1 for streaming
3. **Precision**: Use bfloat16 instead of float32
4. **Clear Cache**: Reset streaming state between conversations

## Evaluation

### FullDuplexBench Integration

PersonaPlex is evaluated on FullDuplexBench, which tests:

1. **User Interruption**: Can handle being interrupted mid-sentence
2. **Pause Handling**: Appropriate pausing and turn-taking
3. **Backchannel**: Natural "uh-huh", "yeah" responses
4. **Smooth Turn Taking**: Minimal awkward gaps/overlaps

**Recommended Prompts**:
- User Interruption: Use assistant prompt
- Others: Use "You enjoy having a good conversation."

### Offline Evaluation

```bash
# Generate response to test audio
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --text-prompt "$(cat prompt.txt)" \
  --input-wav "test_input.wav" \
  --seed 42 \
  --output-wav "output.wav" \
  --output-text "output.json"

# Analyze output
# - Check text transcript quality
# - Measure response latency
# - Evaluate voice consistency
# - Test interruption handling
```

## Installation & Setup

### Prerequisites

**System Requirements**:
- Python 3.10+
- CUDA-capable GPU (recommended: 20GB+ VRAM)
- Linux, macOS, or Windows with WSL

**Audio Codec**:
```bash
# Ubuntu/Debian
sudo apt install libopus-dev

# Fedora/RHEL
sudo dnf install opus-devel

# macOS
brew install opus
```

### Installation Steps

1. **Clone Repository**:
```bash
git clone https://github.com/NVIDIA/personaplex.git
cd personaplex
```

2. **Install Python Package**:
```bash
pip install moshi/.
```

3. **Blackwell GPU (if applicable)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

4. **HuggingFace Authentication**:
   - Visit https://huggingface.co/nvidia/personaplex-7b-v1
   - Accept model license
   - Set environment variable:
```bash
export HF_TOKEN=<your_token>
```

### Running the Server

**Basic**:
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**With CPU Offload**:
```bash
pip install accelerate
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

**Custom Configuration**:
```bash
python -m moshi.server \
  --host localhost \
  --port 8998 \
  --device cuda \
  --voice-prompt-dir /path/to/voices \
  --ssl /path/to/ssl/certs
```

**Access**: Navigate to `https://localhost:8998` in browser

## Code Structure

### Python Package Layout

```
moshi/
├── __init__.py          # Package initialization
├── server.py            # WebSocket server entry point
├── offline.py           # Batch processing entry point
├── client_utils.py      # Logging and utilities
├── models/
│   ├── __init__.py
│   ├── lm.py            # LMModel, LMGen, token sampling
│   ├── compression.py   # MimiModel, audio codec
│   └── loaders.py       # Model loading and initialization
├── modules/
│   ├── conv.py          # Convolutional layers
│   ├── transformer.py   # Transformer implementation
│   ├── seanet.py        # SEANet encoder/decoder
│   ├── streaming.py     # Streaming state management
│   ├── rope.py          # Rotary positional embeddings
│   └── gating.py        # Gating mechanisms
├── quantization/
│   ├── base.py          # Quantizer interface
│   ├── vq.py            # Vector quantization
│   └── core_vq.py       # Core VQ operations
└── utils/
    ├── autocast.py      # Mixed precision utilities
    ├── compile.py       # CUDA graph wrapper
    ├── connection.py    # SSL/TLS helpers
    ├── logging.py       # Colored logging
    └── sampling.py      # Token sampling functions
```

### Client Layout

```
client/
├── src/
│   ├── app.tsx              # Application entry point
│   ├── index.css            # Global styles
│   ├── env.ts               # Environment config
│   ├── audio-processor.ts   # AudioWorklet processor
│   ├── components/
│   │   └── Button/          # Reusable UI components
│   ├── decoder/
│   │   └── decoderWorker.ts # Opus decoding in worker
│   ├── pages/
│   │   ├── Queue/
│   │   │   └── Queue.tsx    # Landing/config page
│   │   └── Conversation/
│   │       └── Conversation.tsx  # Active conversation UI
│   ├── protocol/
│   │   └── types.ts         # WebSocket message types
│   └── modules.d.ts         # TypeScript declarations
├── public/                  # Static assets
├── package.json             # Dependencies
├── tsconfig.json            # TypeScript config
├── vite.config.ts           # Vite build config
└── tailwind.config.js       # Tailwind CSS config
```

## Key Algorithms

### Token Delay Compensation

**Problem**: Autoregressive generation requires previous tokens, but we need causal processing.

**Solution**: Delay pattern allows some tokens to "look ahead"

```python
delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

# At time step t, we can use tokens from step t - delay[k] for stream k
# This allows the model to see "future" user input while generating output
```

**Implementation**:
```python
def _delay_sequence(delays, tensor, padding):
    """Apply delays to token sequences"""
    for k, delay in enumerate(delays):
        tensor[:, k] = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            tensor[:, k, :delay] = padding[:, k]
    return tensor

def _undelay_sequence(delays, tensor):
    """Remove delays to get aligned predictions"""
    for k, delay in enumerate(delays):
        tensor[:, k] = tensor[:, k].roll(-delay, dims=1)
    return tensor
```

### Temperature-based Sampling

**Purpose**: Control randomness in token generation

```python
def sample_token(logits, temperature, top_k=0):
    """
    Sample token from logits with temperature and top-k filtering
    
    Args:
        logits: [batch, vocab] unnormalized log probabilities
        temperature: Controls randomness (0 = greedy, >1 = more random)
        top_k: Only sample from top k tokens (0 = no filtering)
    
    Returns:
        sampled_tokens: [batch] sampled token indices
    """
    # Apply temperature
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        # Set non-top-k logits to -inf
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, top_k_indices, top_k_logits)
    
    # Sample from distribution
    probs = torch.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    
    return sampled
```

**Parameters in PersonaPlex**:
- Audio: temp=0.8, top_k=250 (more variation for natural speech)
- Text: temp=0.7, top_k=25 (more focused for coherent text)

### Voice Prompt Encoding

**Process**:

1. **Load Audio**:
```python
audio = load_audio("voice.wav", sample_rate=24000)
# Shape: [1, samples]
```

2. **Encode to Codes**:
```python
codes = mimi.encode(audio)
# Shape: [1, 8, frames]
```

3. **Feed Through Model**:
```python
lm_gen.streaming_forever(1)
for frame_idx in range(codes.shape[-1]):
    frame = codes[:, :, frame_idx:frame_idx+1]
    embeddings = lm_gen.step(frame, return_embeddings=True)
```

4. **Save Embeddings**:
```python
torch.save(lm_gen.voice_prompt_embeddings, "voice.pt")
```

**Advantage**: Pre-computed embeddings load ~100x faster than encoding audio

## Common Use Cases

### Use Case 1: Q&A Assistant

**Configuration**:
- Voice: NATF2 or NATM1 (natural voices)
- Text Prompt: "You are a wise and friendly teacher..."

**Best For**:
- Educational content
- Information retrieval
- General help and advice

### Use Case 2: Customer Service

**Configuration**:
- Voice: Any natural voice
- Text Prompt: Include company name, role, specific information

**Best For**:
- Appointment scheduling
- Product information
- Support ticket creation
- Order status inquiries

**Example**:
```
You work for TechSupport Pro which is a technical support company 
and your name is Jordan Smith. Information: Handle software 
installation issues for ProductX. Current version is 2.5.1. 
Known issue: antivirus may block installer. Solution: temporarily 
disable antivirus or add exception for C:\Program Files\ProductX.
```

### Use Case 3: Conversational Agent

**Configuration**:
- Voice: Any (experiment with variety for different personalities)
- Text Prompt: "You enjoy having a good conversation." + optional topic

**Best For**:
- Companionship
- Language practice
- Entertainment
- Research on dialogue systems

### Use Case 4: Creative Scenarios

**Configuration**:
- Voice: Match to character
- Text Prompt: Detailed scenario with urgency/stakes

**Best For**:
- Gaming NPCs
- Interactive storytelling
- Training simulations
- Stress testing model capabilities

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:
1. Enable CPU offload:
```bash
python -m moshi.server --cpu-offload
```

2. Use smaller batch size (already 1 by default)

3. Clear CUDA cache between conversations:
```python
model.reset_streaming()
torch.cuda.empty_cache()
```

4. Use float16 instead of bfloat16 (less stable):
```python
lm_model = loaders.get_moshi_lm(
    filename="model.safetensors",
    dtype=torch.float16
)
```

### Issue: Slow Generation

**Solutions**:
1. Ensure CUDA graphs are enabled (automatic on GPU)

2. Check GPU utilization:
```bash
nvidia-smi -l 1
```

3. Verify no CPU fallback:
```python
assert model.device.type == 'cuda', "Model not on GPU!"
```

4. Disable debug checks:
```python
lm_gen = LMGen(lm_model, check=False, report_loss=False)
```

### Issue: Poor Audio Quality

**Solutions**:
1. Check sample rate matches (24kHz)

2. Verify microphone quality and settings

3. Test with pre-recorded high-quality audio

4. Adjust Opus encoding bitrate in client

5. Try different voice prompts

### Issue: Unnatural Responses

**Solutions**:
1. Refine text prompt (be more specific)

2. Adjust sampling temperature:
   - Lower (0.6-0.7): More conservative, repetitive
   - Higher (0.9-1.0): More creative, risky

3. Try different voice prompts (persona affects behavior)

4. Provide more context in the prompt

### Issue: WebSocket Connection Fails

**Solutions**:
1. Check SSL certificate validity

2. Verify port 8998 is not in use:
```bash
netstat -an | grep 8998
```

3. Check firewall settings

4. Try HTTP instead (remove `--ssl` flag)

5. Check browser console for errors

## Research Context

### Paper Details

**Title**: PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models

**Authors**: Rajarshi Roy, Jonathan Raiman, Sang-gil Lee, Teodor-Dumitru Ene, Robert Kirby, Sungwon Kim, Jaehyeon Kim, Bryan Catanzaro (NVIDIA)

**Published**: arXiv:2602.06053 (February 2026)

**Key Contributions**:
1. Method for joint voice and role control in speech models
2. Architecture for full-duplex conversational AI
3. Training methodology combining synthetic and real data
4. Evaluation on FullDuplexBench

### Related Work

**Moshi** (Kyutai, 2024):
- Base architecture for PersonaPlex
- Original full-duplex speech model
- PersonaPlex adds persona control via fine-tuning

**Helium** (Kyutai, 2025):
- Underlying LLM backbone
- Provides generalization capabilities
- Enables out-of-distribution prompting

**FullDuplexBench**:
- Evaluation benchmark for duplex conversation
- Tests interruption, turn-taking, backchanneling
- PersonaPlex evaluated against this benchmark

## Licensing

### Code License

**MIT License** (permissive)
- Free to use, modify, distribute
- Commercial use allowed
- No warranty

**File**: `LICENSE-MIT`

### Model Weights License

**NVIDIA Open Model License**
- Allows research and commercial use
- See HuggingFace model card for full terms
- Accept license on HF to download weights

### Attribution

When using PersonaPlex, cite:

```bibtex
@misc{roy2026personaplexvoicerolecontrol,
      title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models}, 
      author={Rajarshi Roy and Jonathan Raiman and Sang-gil Lee and Teodor-Dumitru Ene and Robert Kirby and Sungwon Kim and Jaehyeon Kim and Bryan Catanzaro},
      year={2026},
      eprint={2602.06053},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06053}, 
}
```

## Advanced Topics

### Custom Training

PersonaPlex is a fine-tune of Moshi. To create your own:

1. Prepare conversation dataset with:
   - Audio pairs (user + assistant)
   - Text prompts defining roles
   - Voice characteristics

2. Use the training code (not included in this repo)

3. Fine-tune from Moshi checkpoint

4. Evaluate on FullDuplexBench

### Integration with Other Systems

**As a Service**:
```python
# Wrap server in REST API
@app.post("/generate")
async def generate(audio: bytes, text_prompt: str, voice: str):
    # Save audio temporarily
    # Call offline.py or use server connection
    # Return result
```

**Telephony Integration**:
- Use Opus codec (already supported)
- Connect via SIP/RTP bridges
- Handle call state management

**Multi-modal Extensions**:
- Add video analysis (face, gestures)
- Incorporate screen sharing context
- Extend prompt with visual information

### Performance Benchmarking

**Latency Measurement**:
```python
import time

# Per-frame latency
start = time.time()
codes = mimi.encode(audio_frame)
tokens = lm_gen.step(codes)
output = mimi.decode(tokens)
latency = time.time() - start
print(f"Frame latency: {latency*1000:.1f}ms")

# Target: <80ms per frame (real-time at 12.5Hz)
```

**Throughput Measurement**:
```python
# Generate 10 seconds of audio
num_frames = int(10 * 12.5)
start = time.time()
for _ in range(num_frames):
    # ... generation loop ...
throughput = num_frames / (time.time() - start)
print(f"Throughput: {throughput:.1f} frames/sec")

# Target: >12.5 frames/sec for real-time
```

## Future Directions

**Potential Enhancements**:
1. Multi-lingual support (current: English only)
2. Emotion control in voice generation
3. Lower latency (higher frame rate)
4. Reduced memory footprint
5. Streaming text output (current: per-token)
6. Multi-speaker conversations
7. Background noise robustness
8. Long-context conversations (>3000 tokens)

## Summary for AI Agents

**Key Takeaways**:

1. **Three-Component System**: Mimi codec + LMModel + LMGen
2. **Dual Control**: Text prompts (persona) + Voice prompts (characteristics)
3. **Full-Duplex**: Both parties can speak simultaneously
4. **Real-Time**: 80ms latency per frame, 12.5Hz frame rate
5. **Flexible Deployment**: Server (real-time) or offline (batch)
6. **Production-Ready**: Docker support, web client, CUDA optimized
7. **Research-Backed**: Published paper, evaluated on benchmarks
8. **Open Source**: MIT license code, downloadable weights

**When to Use PersonaPlex**:
- Building voice assistants with specific personas
- Customer service automation with role-playing
- Research on full-duplex conversation
- Applications requiring natural turn-taking and interruption handling

**When NOT to Use PersonaPlex**:
- Non-English languages (not supported)
- Text-only applications (overkill, use standard LLM)
- Batch transcription (use Whisper or similar)
- Low-resource environments (<8GB GPU memory)

---

**Document Version**: 1.0  
**Last Updated**: 2026-03-01  
**Maintained By**: Roy Holzem (royholzem)  
**Purpose**: Comprehensive guide for AI agents to understand and work with PersonaPlex
