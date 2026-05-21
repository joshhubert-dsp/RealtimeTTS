from RealtimeTTS import ChatterboxEngine, ChatterboxVoice
from RealtimeTTS import ChatterboxFullEngine, ChatterboxFullVoice
from RealtimeTTS import TextToAudioStream

LOREM_IPSUM= "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse et aliquet est. Suspendisse sed fermentum ex. Vivamus lacinia mauris at ex egestas suscipit. Nullam hendrerit eget ante eu tincidunt. "
REFERENCE_AUDIO = "tests/don_sample.wav"

def test_turbo():
    voice = ChatterboxVoice(audio_prompt_path=REFERENCE_AUDIO)
    engine = ChatterboxEngine(device="cuda", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed(LOREM_IPSUM)
    stream.play(
        fast_sentence_fragment=False,
        comma_silence_duration=0.25,
        sentence_silence_duration=0.75,
        debug=True,
    )
    engine.shutdown()

def test_full():
    voice = ChatterboxFullVoice(audio_prompt_path=REFERENCE_AUDIO)
    engine = ChatterboxFullEngine(device="cuda", voice=voice)
    stream = TextToAudioStream(engine)
    stream.feed(LOREM_IPSUM)
    stream.play(
        fast_sentence_fragment=False,
        comma_silence_duration=0.25,
        sentence_silence_duration=0.75,
        debug=True,
    )
    engine.shutdown()

if __name__ == "__main__":
    # test_turbo()
    test_full()