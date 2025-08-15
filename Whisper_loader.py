from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    return processor, model
