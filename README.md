# local-voice

> **Note**: This project from early 2024 is largely obsolete. It was done out of technical curiosity and I didn't have a practical use for it. For current voice assistant projects, consider using newer speech to speech models as a starting point.

An implementation of a Voice Activity Detection -> Speech to Text -> LLM -> Text to Speech pipeline using only local mlx and onnx models. This will only work on Apple devices due to the use of mlx. You could build something similar using whisper.cpp and llama.cpp that should work cross device.

The speech to speech times are ok for basic conversation. The main challenge with making it more useful is reducing speech to text errors, especially on short utterances. The pipeline approach is probably obsolete with upcoming multimodal models, although it might have some advantages, such as easier visibility into what was heard.

Features:
* Speech to speech times in 500ms to 1500ms range on a M2 Max
  * Within conversational speech latency
  * Using Whisper large v3 + Llama 3 8B (both with mlx 4 bit quantization)
* Interuptible, will stop talking as soon as new speech has been heard
* Should be possible to make it multilingual
  * Whisper model is multilingual
  * Llama 3 understands multiple languages
  * Piper has voices available in different languages
* Relatively minimal dependencies, the models are doing most of the work
* Various logs are saved, so you can collect audio and transcription data for model evaluation or improvements

Caveats:
* The codebase is fairly hacky, so might it might require some work to get it working for you
* Doesn't filter out its own voice, so will only work when using a headset or speakers which include echo cancellation
* Doesn't work with multiple speakers, the LLM will get confused with two separate people talking
* Works best if you manually set an audio prefix using the notebooks/record_audio_prefix.ipynb notebook
* You can interupt it by saying "Stop!" but it will then respond to your message ("Ok, I'll stop.") rather than stop talking entirely

## Overview

Architecture:
* Uses a crude Actor model to separate out components and handle communication between them using pubsub
* Python asyncio to interleave model computations between components
  * For example LLM decoding will pause after every token to allow STT to check if there is an interruption

VAD (Voice Activity Detection)
* Uses the webrtcvad implementation of webrtc vad algorithm
* Batches incoming audio into segments of 100ms for vad checks
* Should be tuned for your environment, set to level 2 (on scale of 1, 2, 3)
* VAD level is optimized for recall over precision, so there is a lot wasted computation by Whisper transcribing empty audio
 * Especially when there is background noise like typing

Speech to Text
* Whisper mlx implementation
* Dynamically reduces the size of the audio encoder to match audio length
  * Found that this didn't seem to impact accuracy on audio clips >5 seconds long, evaluated on fleurs dataset
  * Significantly improves latency
* Uses an audio clip with a known transcription to improve accuracy on short utterances (like "Hello" or "How are you?")
  * This audio clip is prepended to the new audio and decoding assumes the provided transcription is correct, rather than sampling
  * Evaluation in notebooks/audio_prefix_analysis.ipynb suggests this works best when it's recorded with your voice and microphone
  * The clip is collected automatically if an audio prefix is not provided
  * This technique seemed to reduce the problems of hallucination on audio without speech
* Word Error Rates for short utterances were around 15% using audio prefix compared to 30% without
  * Subjectively seemed a lot better for longer speech
  * The error rate is low enough that you can have a short conversation, but not good enough for something like reliable voice commands

LLM
* Llama 3 8B instruct model
  * Works well for simple conversations, can ask some interesting questions and get good responses
* KV caching is used to keep track of the conversation, so only minimal updates are needed for each new segment of transcribed speech
* Uses the model to estimate when the user has finished speaking
  * If the most likely next token is the end of user segment token, the model will start generating a response
* Generates response tokens in chunks, pausing to allow other components in the pipeline to process
* Sends response text asap to speech component, stopping whenever punctuation is detected
* Prompting encourages a short initial phrase (like "Ok," or "Sorry,") so a first word or short phrase is spoken asap in response
* There is a lot of code to track conversational state
  * You could probably remove this by fine tuning the logic into the LLM

Text to Speech
* Rhasspy Piper model
  * Works great, this was previously one of the slowest parts of the pipeline, now speech generation usually within 100ms
* Depends upon espeak for converting text to phonemes
  * Assumed to be found at ```/opt/homebrew/bin/espeak```
* A minimal version of the onnx inference is included in min_rhasspy_piper/voice.py so there is no direct dependency on the rhasspy/piper project
* Interruptions immediately stop speech generation, without the LLM being involved in that decision
  * Speech audio is played out in segments of 100ms, the queue is cleared out immediately on interruption


A typical speech to speech latency breakdown looks like:
* STT: ~300ms
* LLM: ~400ms
* TTS: ~100ms 

## Setup

Install homebrew dependencies, espeak for the piper voice model and portaudio for pyaudio:

```
brew install espeak portaudio
```

Install Python requirements

```
pip install -r requirements.txt
```

The setup_models.sh script will download the Whisper and Piper models into a local directory

```
# You can make MODEL_DIR somewhere else if preferred
MODEL_DIR="models"
setup_models.sh $MODEL_DIR
```

## Usage

Include ```TOKENIZERS_PARALLELISM=true``` to suppress a huggingface tokenizer parallelism error.

```
TOKENIZERS_PARALLELISM=true python main.py \
    --whisper_mlx_model_dir $MODEL_DIR/whisper  \
    --mlxlm_model_path mlx-community/Meta-Llama-3-8B-Instruct-4bit  \
    --piper_model_dir $MODEL_DIR/piper  \
    --log_level INFO 2>&1 | tee logs/main.log
```

Output should look similar to:

```
2024-07-26 23:42:43,551 - root - INFO - Initializing ...
2024-07-26 23:42:44,609 - root - INFO - Starting input audio ...
2024-07-26 23:42:44,609 - audio_io - INFO - Starting audio input
2024-07-26 23:30:21,786 - incremental_transcriber - INFO - whisper: 238ms :  How does a hurricane start? (0.63)
2024-07-26 23:30:22,217 - llm_chat - INFO - Speaking response: Over warm ocean water,
2024-07-26 23:30:22,300 - speech - INFO - Speech generation took 83 milliseconds
2024-07-26 23:30:22,300 - __main__ - INFO - Speech to speech ms: 856
2024-07-26 23:30:22,424 - llm_chat - INFO - Speaking response:  usually in the Caribbean or Atlantic.
2024-07-26 23:30:22,528 - speech - INFO - Speech generation took 104 milliseconds
2024-07-26 23:30:22,533 - llm_chat - INFO - Speaking response: 
2024-07-26 23:30:22,533 - speech - INFO - Speech generation took 0 milliseconds
2024-07-26 23:30:34,599 - incremental_transcriber - INFO - New audio prefix: logprob=-0.2633504867553711 tokens=[1664, 291, 2903, 577, 309, 1985, 30]
2024-07-26 23:30:34,622 - incremental_transcriber - INFO - whisper: 254ms :  Can you explain how it works? (0.77)
2024-07-26 23:30:35,222 - llm_chat - INFO - Speaking response: It's a big storm system that forms when warm air rises,
2024-07-26 23:30:35,366 - speech - INFO - Speech generation took 144 milliseconds
2024-07-26 23:30:35,367 - __main__ - INFO - Speech to speech ms: 1102
2024-07-26 23:30:35,420 - llm_chat - INFO - Speaking response:  cools,
2024-07-26 23:30:35,467 - speech - INFO - Speech generation took 46 milliseconds
2024-07-26 23:30:35,537 - llm_chat - INFO - Speaking response:  and condenses,
2024-07-26 23:30:35,605 - speech - INFO - Speech generation took 68 milliseconds
2024-07-26 23:30:35,729 - llm_chat - INFO - Speaking response:  creating strong winds and heavy rain.
2024-07-26 23:30:35,835 - speech - INFO - Speech generation took 106 milliseconds
2024-07-26 23:30:35,843 - llm_chat - INFO - Speaking response: 
2024-07-26 23:30:35,843 - speech - INFO - Speech generation took 0 milliseconds
2024-07-26 23:31:06,545 - incremental_transcriber - INFO - whisper: 281ms :  How do you measure the strength of a hurricane? (0.75)
2024-07-26 23:31:07,048 - llm_chat - INFO - Speaking response: By its wind speed,
2024-07-26 23:31:07,122 - speech - INFO - Speech generation took 74 milliseconds
2024-07-26 23:31:07,123 - __main__ - INFO - Speech to speech ms: 962
2024-07-26 23:31:07,227 - llm_chat - INFO - Speaking response:  usually in miles per hour,
2024-07-26 23:31:07,321 - speech - INFO - Speech generation took 93 milliseconds
2024-07-26 23:31:07,494 - llm_chat - INFO - Speaking response:  using the Saffir-Simpson scale.
2024-07-26 23:31:07,584 - speech - INFO - Speech generation took 90 milliseconds
2024-07-26 23:31:07,589 - llm_chat - INFO - Speaking response: 
2024-07-26 23:31:07,589 - speech - INFO - Speech generation took 0 milliseconds
2024-07-26 23:31:20,628 - incremental_transcriber - INFO - whisper: 272ms :  What is the strongest hurricane ever recorded? (0.72)
2024-07-26 23:31:21,081 - llm_chat - INFO - Speaking response: Hurricane Patricia,
2024-07-26 23:31:21,155 - speech - INFO - Speech generation took 73 milliseconds
2024-07-26 23:31:21,155 - __main__ - INFO - Speech to speech ms: 903
2024-07-26 23:31:21,278 - llm_chat - INFO - Speaking response:  it was a Category 5.
2024-07-26 23:31:21,358 - speech - INFO - Speech generation took 79 milliseconds
2024-07-26 23:31:21,363 - llm_chat - INFO - Speaking response: 
2024-07-26 23:31:21,363 - speech - INFO - Speech generation took 0 milliseconds
2024-07-26 23:31:28,628 - incremental_transcriber - INFO - whisper: 167ms :  (0.83)
```

## Acknowledgements

This project wouldn't be possible without the underlying models and MLX:

* [Whisper](https://github.com/openai/whisper)
* [mlx](https://github.com/ml-explore/mlx)
* [rhasspy piper](https://github.com/rhasspy/piper)
* [Llama](https://github.com/facebookresearch/llama)