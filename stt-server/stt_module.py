from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
from pydub import AudioSegment, effects
import os
import gc
import concurrent.futures
from tqdm import tqdm

def convert_to_wav(input_path: str) -> str:
    """Convert input audio file to WAV format."""
    audio = AudioSegment.from_file(input_path)
    audio = effects.normalize(audio)
    output_path = os.path.splitext(input_path)[0] + ".wav"
    audio.export(output_path, format="wav")
    return output_path

def split_audio(audio_path: str, chunk_length_ms: int = 15000, overlap_ms: int = 1000) -> list:
    """Split WAV audio into overlapping chunks."""
    audio = AudioSegment.from_wav(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms - overlap_ms):
        chunk = audio[i:i + chunk_length_ms]
        if len(chunk) < 1000:
            break
        chunk_path = f"{os.path.splitext(audio_path)[0]}_chunk_{len(chunks)}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def process_chunk(chunk_path: str, pipe) -> str:
    """Run STT on a single chunk."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    try:
        result = pipe(
            chunk_path,
            return_timestamps=True,
            chunk_length_s=15,
            stride_length_s=1,
            batch_size=1
        )
        return result.get("text", "").strip()
    except Exception:
        return ""

def transcribe_long_audio(file_path: str, model_name: str = "openai/whisper-large-v2") -> str:
    """Full STT pipeline from audio file to final transcript."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.backends.cudnn.benchmark = True

    wav_path = convert_to_wav(file_path)
    device = 0 if torch.cuda.is_available() else -1

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )
    processor = WhisperProcessor.from_pretrained(model_name)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        model_kwargs={"use_cache": True}
    )

    chunks = split_audio(wav_path)
    results = []

    for i in range(0, len(chunks), 5):
        batch_chunks = chunks[i:i + 5]
        batch_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk, pipe): idx
                for idx, chunk in enumerate(batch_chunks)
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_chunk), total=len(batch_chunks)):
                idx = future_to_chunk[future]
                try:
                    batch_results.append((idx, future.result()))
                except:
                    batch_results.append((idx, ""))

        batch_results.sort(key=lambda x: x[0])
        results.extend([r[1] for r in batch_results])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Cleanup
    for chunk in chunks:
        try:
            os.remove(chunk)
        except:
            pass

    return " ".join(results)

