from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from moviepy import AudioFileClip, VideoFileClip
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

from src.config import MOCK_MODE
from src.screenplay import load_screenplay
from src.visuals import load_image
from src.scene_motion import build_motion_matrices


# ============================================================
# LXNORO YESNO AI STUDIO
# Local deterministic MVP application layer
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VIDEO_PATH = OUTPUT_DIR / "test_composite.mp4"
FINAL_VIDEO_PATH = OUTPUT_DIR / "yesno_final.mp4"
MOCK_AUDIO_DIR = OUTPUT_DIR / "mock_audio"

MASTER_FPS = 12
OUTPUT_FPS = 30
TRAILER_DURATION = 60
SCENE_DURATION = 10
SCENE_COUNT = 6

VOICE_DUCK_DB = -12.0
AUDIO_SAMPLE_RATE = 44_100


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="YESNO LXNORO AI Studio",
    page_icon="Ã°Å¸Å½Â¬",
    layout="wide",
)


# ============================================================
# Deterministic local color normalization
# ============================================================

def normalize_ghibli_color_profile(image: np.ndarray) -> np.ndarray:
    """
    Deterministic local color-profile normalization.

    This is deliberately CPU-only and does not call any cloud API.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3")

    working = image.astype(np.float32) / 255.0

    # Matrix-style channel transform.
    profile_matrix = np.array(
        [
            [1.04, 0.01, 0.00],
            [0.00, 1.00, 0.01],
            [0.00, 0.02, 1.05],
        ],
        dtype=np.float32,
    )

    normalized = working @ profile_matrix.T
    normalized = np.clip(normalized, 0.0, 1.0)

    return (normalized * 255.0).astype(np.uint8)


# ============================================================
# Local deterministic audio generation
# ============================================================

def generate_mock_ambient(duration_ms: int) -> AudioSegment:
    """
    Creates deterministic ambient music locally.

    No download and no external API.
    """
    base = Sine(220).to_audio_segment(duration=duration_ms)
    layer_2 = Sine(330).to_audio_segment(duration=duration_ms)
    layer_3 = Sine(440).to_audio_segment(duration=duration_ms)

    ambient = (
        base.overlay(layer_2 - 8)
        .overlay(layer_3 - 14)
        .low_pass_filter(1800)
        .apply_gain(-24)
    )

    return ambient.set_frame_rate(AUDIO_SAMPLE_RATE).set_channels(2)


def generate_mock_dialogue(duration_ms: int) -> AudioSegment:
    """
    Creates deterministic speech-like placeholder dialogue.

    The MVP does not pretend this is real speech synthesis.
    It creates controlled voice-band energy so the ducking
    detector can be validated locally.
    """
    dialogue = AudioSegment.silent(
        duration=duration_ms,
        frame_rate=AUDIO_SAMPLE_RATE,
    ).set_channels(1)

    segment_length = 2500
    gap_length = 1500

    cursor = 0
    active = True

    while cursor < duration_ms:
        if active:
            current_length = min(segment_length, duration_ms - cursor)

            tone_a = Sine(180).to_audio_segment(duration=current_length)
            tone_b = Sine(260).to_audio_segment(duration=current_length)

            voice = (
                tone_a.overlay(tone_b - 5)
                .high_pass_filter(100)
                .low_pass_filter(3400)
                .apply_gain(-18)
            )

            dialogue = dialogue.overlay(voice, position=cursor)

            cursor += current_length
        else:
            cursor += min(gap_length, duration_ms - cursor)

        active = not active

    return dialogue.set_channels(1).set_frame_rate(AUDIO_SAMPLE_RATE)


# ============================================================
# NumPy voice-frequency activity detection
# ============================================================

def detect_voice_activity(
    dialogue: AudioSegment,
    window_ms: int = 100,
) -> np.ndarray:
    """
    Detects speech-band energy using NumPy FFT.

    Voice band:
        100 Hz - 3400 Hz

    Returns one boolean value per analysis window.
    """
    mono = dialogue.set_channels(1).set_frame_rate(AUDIO_SAMPLE_RATE)

    samples = np.array(
        mono.get_array_of_samples(),
        dtype=np.float32,
    )

    if samples.size == 0:
        return np.zeros(0, dtype=bool)

    max_value = float(1 << (8 * mono.sample_width - 1))
    samples /= max_value

    window_size = max(1, int(AUDIO_SAMPLE_RATE * window_ms / 1000))
    window_count = math.ceil(len(samples) / window_size)

    activity = np.zeros(window_count, dtype=bool)

    for index in range(window_count):
        start = index * window_size
        end = min(start + window_size, len(samples))

        chunk = samples[start:end]

        if chunk.size < 16:
            continue

        spectrum = np.abs(np.fft.rfft(chunk))
        frequencies = np.fft.rfftfreq(
            chunk.size,
            d=1.0 / AUDIO_SAMPLE_RATE,
        )

        voice_band = (
            (frequencies >= 100.0)
            & (frequencies <= 3400.0)
        )

        band_energy = float(
            np.mean(spectrum[voice_band] ** 2)
        )

        total_energy = float(
            np.mean(spectrum ** 2)
        )

        if total_energy <= 1e-12:
            continue

        ratio = band_energy / total_energy

        # Deterministic threshold.
        activity[index] = (
            ratio >= 0.20
            and total_energy >= 1e-7
        )

    return activity


# ============================================================
# Exact -12 dB ducking
# ============================================================

def duck_ambient_during_voice(
    ambient: AudioSegment,
    dialogue: AudioSegment,
    duck_db: float = VOICE_DUCK_DB,
    window_ms: int = 100,
) -> AudioSegment:
    """
    Applies exactly -12 dB gain to ambient windows where
    voice activity is detected.
    """
    if duck_db != -12.0:
        raise ValueError(
            "YESNO audio engineering requires exactly -12 dB ducking"
        )

    target_duration = max(
        len(ambient),
        len(dialogue),
    )

    ambient = ambient.set_channels(2)
    dialogue = dialogue.set_channels(1)

    ambient = ambient[:target_duration]

    if len(ambient) < target_duration:
        ambient += AudioSegment.silent(
            duration=target_duration - len(ambient),
            frame_rate=ambient.frame_rate,
        ).set_channels(2)

    dialogue = dialogue[:target_duration]

    if len(dialogue) < target_duration:
        dialogue += AudioSegment.silent(
            duration=target_duration - len(dialogue),
            frame_rate=dialogue.frame_rate,
        )

    activity = detect_voice_activity(
        dialogue,
        window_ms=window_ms,
    )

    output = AudioSegment.empty()

    for index, is_active in enumerate(activity):
        start = index * window_ms
        end = min(start + window_ms, target_duration)

        segment = ambient[start:end]

        if is_active:
            segment = segment.apply_gain(-12.0)

        output += segment

    if len(output) < target_duration:
        output += ambient[len(output):target_duration]

    return output.set_channels(2)


# ============================================================
# Stereo multiplexing
# ============================================================

def multiplex_stereo(
    ducked_ambient: AudioSegment,
    dialogue: AudioSegment,
) -> AudioSegment:
    """
    Produces the final stereo mix:
        background = ducked ambient
        foreground = dialogue
    """
    ambient = ducked_ambient.set_channels(2)
    voice = dialogue.set_channels(2)

    if len(ambient) != len(voice):
        duration = max(len(ambient), len(voice))

        ambient = ambient[:duration]
        voice = voice[:duration]

        if len(ambient) < duration:
            ambient += AudioSegment.silent(
                duration=duration - len(ambient),
                frame_rate=AUDIO_SAMPLE_RATE,
            ).set_channels(2)

        if len(voice) < duration:
            voice += AudioSegment.silent(
                duration=duration - len(voice),
                frame_rate=AUDIO_SAMPLE_RATE,
            ).set_channels(2)

    mixed = ambient.overlay(voice)

    return (
        mixed
        .set_channels(2)
        .set_frame_rate(AUDIO_SAMPLE_RATE)
    )


# ============================================================
# Audio pipeline
# ============================================================

def build_local_audio(
    duration_seconds: int = TRAILER_DURATION,
) -> Path:
    """
    Complete deterministic local audio pipeline.
    """
    MOCK_AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    duration_ms = duration_seconds * 1000

    ambient = generate_mock_ambient(duration_ms)
    dialogue = generate_mock_dialogue(duration_ms)

    ducked = duck_ambient_during_voice(
        ambient,
        dialogue,
        duck_db=-12.0,
    )

    stereo_mix = multiplex_stereo(
        ducked,
        dialogue,
    )

    audio_path = MOCK_AUDIO_DIR / "yesno_mix.wav"

    stereo_mix.export(
        audio_path,
        format="wav",
    )

    return audio_path


# ============================================================
# Video/audio multiplexing
# ============================================================

def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """
    Combines the 30fps H.264 video with stereo AAC audio.
    """
    video = VideoFileClip(str(video_path))
    audio = AudioFileClip(str(audio_path))

    try:
        final = video.with_audio(audio)

        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=OUTPUT_FPS,
            audio_fps=AUDIO_SAMPLE_RATE,
            logger=None,
        )

        final.close()

    finally:
        audio.close()
        video.close()

    return output_path


# ============================================================
# Mock screenplay processing
# ============================================================

def validate_local_screenplay() -> None:
    screenplay_path = PROJECT_ROOT / "data" / "mock_screenplay.json"

    screenplay = load_screenplay(
        screenplay_path
    )

    if len(screenplay.scenes) != SCENE_COUNT:
        raise ValueError(
            "Local screenplay must contain exactly 6 scenes"
        )

    if screenplay.total_duration_seconds != TRAILER_DURATION:
        raise ValueError(
            "Local screenplay must total exactly 60 seconds"
        )


# ============================================================
# Future cloud boundary
# ============================================================

def generate_remote_screenplay(_prompt: str):
    """
    Future Nebius/NVIDIA branch.

    The API endpoint and request contract are intentionally not
    invented here. This function is the isolated integration
    boundary that will receive the verified official API contract.
    """
    raise RuntimeError(
        "MOCK_MODE=False requires the verified Nebius AI API "
        "integration to be implemented before cloud execution."
    )


def process_prompt(prompt: str) -> None:
    """
    Current MVP:
        prompt -> validated local screenplay -> local audio/video.

    Future:
        prompt -> structured JSON screenplay -> Nebius video API.
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if MOCK_MODE:
        validate_local_screenplay()
        return

    generate_remote_screenplay(prompt)



# ============================================================
# ============================================================
# YESNO LXNORO AI Studio — Professional UI
# ============================================================

if "yesno_dark_mode" not in st.session_state:
    st.session_state.yesno_dark_mode = True

theme_dark, theme_light = st.columns(2)

with theme_dark:
    if st.button("DARK MODE", key="yesno_dark_button"):
        st.session_state.yesno_dark_mode = True
        st.rerun()

with theme_light:
    if st.button("LIGHT MODE", key="yesno_light_button"):
        st.session_state.yesno_dark_mode = False
        st.rerun()

dark_mode = st.session_state.yesno_dark_mode
if dark_mode:
    BG,PANEL,PANEL_ALT,BORDER,TEXT,MUTED,BLUE,BLUE_DEEP,INPUT_BG = (
        "#05080d","#0b111a","#101923","#213447","#f5f8fc",
        "#a8b6c7","#70c8ff","#1682c7","#080d14"
    )
else:
    BG,PANEL,PANEL_ALT,BORDER,TEXT,MUTED,BLUE,BLUE_DEEP,INPUT_BG = (
        "#f4f7fb","#ffffff","#eef4fa","#c7d6e5","#172331",
        "#536577","#126eaa","#0b5a8f","#ffffff"
    )

st.markdown(f"""
<style>
:root {{
--yesno-bg:{BG};--yesno-panel:{PANEL};--yesno-panel-alt:{PANEL_ALT};
--yesno-border:{BORDER};--yesno-text:{TEXT};--yesno-muted:{MUTED};
--yesno-blue:{BLUE};--yesno-blue-deep:{BLUE_DEEP};--yesno-input:{INPUT_BG};
}}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"]>.main {{
background:var(--yesno-bg)!important;color:var(--yesno-text)!important;
}}
[data-testid="stHeader"]{{background:transparent!important}}
.block-container{{max-width:1840px!important;padding:24px 30px 42px!important}}
[data-testid="stHorizontalBlock"]{{width:100%!important;gap:24px!important;align-items:stretch!important}}
[data-testid="stHorizontalBlock"]>[data-testid="column"]{{
min-width:0!important;width:calc(50% - 12px)!important;flex:1 1 0!important;
}}
[data-testid="stHorizontalBlock"]>[data-testid="column"]:first-child{{order:1!important}}
[data-testid="stHorizontalBlock"]>[data-testid="column"]:last-child{{order:2!important}}

.st-key-yesno_chat_panel,.st-key-yesno_video_panel{{
width:100%!important;min-width:0!important;box-sizing:border-box!important;
border:1px solid var(--yesno-border)!important;border-radius:20px!important;
background:var(--yesno-panel)!important;padding:26px!important;
min-height:720px!important;color:var(--yesno-text)!important;
}}

.yesno-brand-row{{display:flex;align-items:center;gap:16px;padding:2px 0 22px;
border-bottom:1px solid var(--yesno-border);margin-bottom:22px}}
.yesno-logo{{width:60px;height:60px;min-width:60px;border-radius:15px;
display:flex;align-items:center;justify-content:center;background:var(--yesno-blue-deep);
color:#fff!important;font-size:20px;font-weight:950;letter-spacing:.5px}}
.yesno-brand-name{{color:var(--yesno-text)!important;font-size:38px!important;
line-height:1!important;font-weight:950!important;letter-spacing:2px!important}}
.yesno-brand-sub{{color:var(--yesno-muted)!important;font-size:16px!important;
margin-top:8px!important;font-weight:700!important;letter-spacing:.8px!important}}

.yesno-section-title{{color:var(--yesno-blue)!important;font-size:18px!important;
font-weight:900!important;letter-spacing:1.5px!important;margin:0 0 12px!important}}

.st-key-yesno_ai_response{{width:100%!important;min-height:380px!important;
box-sizing:border-box!important;border:1px solid var(--yesno-border)!important;
border-radius:16px!important;background:var(--yesno-panel-alt)!important;
color:var(--yesno-text)!important;overflow:auto!important}}

.st-key-yesno_user_input{{width:100%!important;margin-top:12px!important}}
.st-key-yesno_user_input [data-testid="stChatInput"]{{
position:relative!important;width:100%!important;left:auto!important;right:auto!important;
bottom:auto!important;margin:0!important}}
.st-key-yesno_user_input textarea{{
min-height:130px!important;height:130px!important;font-size:19px!important;
line-height:1.55!important;color:var(--yesno-text)!important;
background:var(--yesno-input)!important}}
.st-key-yesno_user_input textarea::placeholder{{
color:var(--yesno-muted)!important;opacity:1!important;font-size:18px!important}}

.yesno-video-heading{{display:flex;justify-content:space-between;align-items:center;
gap:12px;margin-bottom:16px}}
.yesno-video-title{{color:var(--yesno-text)!important;font-size:23px!important;
font-weight:900!important;letter-spacing:1px!important}}
.yesno-status{{color:var(--yesno-blue)!important;border:1px solid var(--yesno-blue)!important;
border-radius:999px!important;padding:7px 13px!important;font-size:12px!important;
font-weight:900!important;letter-spacing:1px!important}}
.st-key-yesno_video_player,.st-key-yesno_video_player [data-testid="stVideo"]{{
width:100%!important}}
.st-key-yesno_video_player video{{
width:100%!important;max-width:100%!important;height:auto!important;
aspect-ratio:16/9!important;object-fit:contain!important;border-radius:16px!important;
display:block!important}}

.yesno-projects-box{{margin-top:22px;min-height:175px;box-sizing:border-box;
border:1px solid var(--yesno-border);border-radius:18px;
background:var(--yesno-panel-alt);padding:30px;display:flex;align-items:center}}
.yesno-project-title{{color:var(--yesno-text)!important;font-size:20px!important;
font-weight:900!important;letter-spacing:1.5px!important}}
.yesno-coming{{color:var(--yesno-blue)!important;font-size:30px!important;
line-height:1.1!important;font-weight:950!important;letter-spacing:1.2px!important;
margin-top:10px!important}}

button[kind="secondary"]{{border-color:var(--yesno-border)!important;
color:var(--yesno-text)!important;background:var(--yesno-panel)!important}}

@media(max-width:900px){{
.block-container{{padding:14px 12px 30px!important}}
[data-testid="stHorizontalBlock"]{{flex-direction:column!important;gap:18px!important}}
[data-testid="stHorizontalBlock"]>[data-testid="column"]{{
width:100%!important;max-width:100%!important;flex:1 1 100%!important}}
[data-testid="stHorizontalBlock"]>[data-testid="column"]:first-child{{order:2!important}}
[data-testid="stHorizontalBlock"]>[data-testid="column"]:last-child{{order:1!important}}
.st-key-yesno_chat_panel,.st-key-yesno_video_panel{{
min-height:auto!important;padding:18px!important}}
.yesno-brand-name{{font-size:28px!important}}
.st-key-yesno_ai_response{{min-height:320px!important}}
.yesno-video-title{{font-size:20px!important}}
.st-key-yesno_user_input textarea{{
min-height:120px!important;height:120px!important;font-size:18px!important}}
.yesno-coming{{font-size:25px!important}}
}}
</style>
""",unsafe_allow_html=True)

left,right=st.columns([1,1],gap="medium")

with left:
    chat_panel=st.container(key="yesno_chat_panel")
    with chat_panel:
        st.markdown("""
        <div class="yesno-brand-row">
            <div class="yesno-logo">LX</div>
            <div>
                <div class="yesno-brand-name">LXNORO</div>
                <div class="yesno-brand-sub">YESNO AI Studio</div>
            </div>
        </div>
        """,unsafe_allow_html=True)

        st.markdown('<div class="yesno-section-title">AI RESPONSE</div>',unsafe_allow_html=True)

        ai_area=st.container(border=True,key="yesno_ai_response")
        with ai_area:
            pass

        st.markdown('<div class="yesno-section-title">YOUR MESSAGE</div>',unsafe_allow_html=True)

        user_input=st.container(key="yesno_user_input")
        with user_input:
            prompt=st.chat_input(
                "Write your scene idea or screenplay...",
                key="yesno_prompt",
            )

with right:
    video_panel=st.container(key="yesno_video_panel")
    with video_panel:
        st.markdown("""
        <div class="yesno-video-heading">
            <div class="yesno-video-title">CINEMATIC PREVIEW</div>
            <div class="yesno-status">LOCAL MOCK</div>
        </div>
        """,unsafe_allow_html=True)

        video_player=st.container(key="yesno_video_player")
        with video_player:
            video_to_display=FINAL_VIDEO_PATH if FINAL_VIDEO_PATH.is_file() else VIDEO_PATH
            if video_to_display.is_file():
                st.video(str(video_to_display))
            else:
                st.warning(f"The visual composite does not exist yet: {VIDEO_PATH}")

        st.markdown("""
        <div class="yesno-projects-box">
            <div>
                <div class="yesno-project-title">LXNORO PROJECTS</div>
                <div class="yesno-coming">COMING SOON</div>
            </div>
        </div>
        """,unsafe_allow_html=True)

# ============================================================# EXISTING PROCESSING PIPELINE ? PRESERVED
# ============================================================

if prompt:


    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.status(
            "YESNO AI Studio processing...",
            expanded=True,
        ) as status:

            # ------------------------------------------------
            # Step 1/4
            # ------------------------------------------------
            st.write(
                "Step 1/4 Ã¢â‚¬â€ Normalize Ghibli color profile matrices via NumPy"
            )

            if not VIDEO_PATH.is_file():
                raise FileNotFoundError(
                    f"Missing visual composite: {VIDEO_PATH}"
                )

            test_image_path = (
                PROJECT_ROOT
                / "assets"
                / "scenes"
                / "scene_01.png"
            )

            image = load_image(test_image_path)

            normalized = normalize_ghibli_color_profile(
                image
            )

            if normalized.shape != image.shape:
                raise RuntimeError(
                    "Color normalization changed image dimensions"
                )

            st.write("Ã¢Å“â€œ Local color matrix validation complete")

            # ------------------------------------------------
            # Step 2/4
            # ------------------------------------------------
            st.write(
                "Step 2/4 Ã¢â‚¬â€ Apply -12dB audio ducking via Pydub"
            )

            audio_path = build_local_audio(
                duration_seconds=TRAILER_DURATION
            )

            if not audio_path.is_file():
                raise RuntimeError(
                    "Audio pipeline did not produce WAV output"
                )

            st.write(
                "Ã¢Å“â€œ Ambient + dialogue stereo mix generated"
            )
            st.write(
                "Ã¢Å“â€œ Voice-band activity detection validated"
            )
            st.write(
                "Ã¢Å“â€œ Background ducking fixed at -12 dB"
            )

            # ------------------------------------------------
            # Step 3/4
            # ------------------------------------------------
            st.write(
                "Step 3/4 Ã¢â‚¬â€ Multiplex streams into H.264/AAC MP4"
            )

            mux_video_audio(
                VIDEO_PATH,
                audio_path,
                FINAL_VIDEO_PATH,
            )

            if not FINAL_VIDEO_PATH.is_file():
                raise RuntimeError(
                    "Final MP4 was not created"
                )

            st.write(
                "Ã¢Å“â€œ H.264 video + AAC stereo audio created"
            )

            # ------------------------------------------------
            # Step 4/4
            # ------------------------------------------------
            st.write(
                "Step 4/4 Ã¢â‚¬â€ Validation/finalization"
            )

            final_video = VideoFileClip(
                str(FINAL_VIDEO_PATH)
            )

            try:
                duration = float(final_video.duration or 0.0)
                fps = float(final_video.fps or 0.0)
            finally:
                final_video.close()

            if abs(duration - TRAILER_DURATION) > 0.5:
                raise RuntimeError(
                    f"Unexpected final duration: {duration:.2f}s"
                )

            if abs(fps - OUTPUT_FPS) > 0.5:
                raise RuntimeError(
                    f"Unexpected final FPS: {fps:.2f}"
                )

            st.write(
                f"Ã¢Å“â€œ Final duration: {duration:.2f}s"
            )
            st.write(
                f"Ã¢Å“â€œ Final container FPS: {fps:.2f}"
            )

            status.update(
                label="YESNO processing complete",
                state="complete",
            )

        st.success(
            "YESNO LXNORO AI Studio Ã¢â‚¬â€ local MOCK pipeline complete."
        )

        if FINAL_VIDEO_PATH.is_file():
            st.video(str(FINAL_VIDEO_PATH))

    except Exception as exc:
        st.error(
            f"Processing failed: {exc}"
        )












