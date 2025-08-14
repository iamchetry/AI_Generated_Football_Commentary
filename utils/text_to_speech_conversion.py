from gtts import gTTS
from elevenlabs.client import ElevenLabs
import ffmpeg
# from pydub import AudioSegment
from utils.key import elevan_labs_key

# AudioSegment.converter = '/opt/homebrew/bin/ffmpeg'  # or your ffmpeg path

eleven = ElevenLabs(api_key=elevan_labs_key)


def tts_google(commentary_text, output_audio_path):
    tts = gTTS(text=commentary_text, lang='en')
    tts.save(output_audio_path)


def tts_elevanlabs(text, output_path, voice_id='pNInz6obpgDQGcFmaJgB'):
    audio_generator = eleven.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings={"stability": 0.3, "style": 0.9}
    )
    
    # Join all chunks from generator into bytes
    audio_bytes = b''.join(audio_generator)
    
    with open(output_path, 'wb') as f:
        f.write(audio_bytes)


# def speed_up_audio(input_path, output_path, speed=None):
#     sound = AudioSegment.from_file(input_path)
#     faster = sound._spawn(sound.raw_data, overrides={
#         "frame_rate": int(sound.frame_rate * speed)
#     }).set_frame_rate(sound.frame_rate)
#     faster.export(output_path, format='mp3')


def speed_up_audio(input_path, output_path, speed=None):
    ffmpeg.input(input_path).output(
        output_path,
        filter_complex=f"atempo={speed}",
        acodec="libmp3lame"
    ).run(overwrite_output=True)
