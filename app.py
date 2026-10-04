from __future__ import annotations

import array
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import torch
from silero_vad import get_speech_timestamps, load_silero_vad


SAMPLE_RATE = 16_000
SUPPORTED_EXTENSIONS = {"mp4", "mov", "mkv"}


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def run_command(args: list[str], error_message: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "沒有取得錯誤訊息。"
        raise RuntimeError(f"{error_message}\n\n{detail[-4000:]}")
    return result


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "找不到 FFmpeg 工具："
            + ", ".join(missing)
            + "。請先安裝 FFmpeg，並確認 ffmpeg / ffprobe 可以在終端機中執行。"
        )


def get_video_duration(video_path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        "無法讀取影片長度。",
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("ffprobe 回傳的影片長度格式不正確。") from exc
    if duration <= 0:
        raise RuntimeError("影片長度必須大於 0 秒。")
    return duration


def extract_mono_wav(video_path: Path, wav_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        "無法從影片抽取單聲道 16kHz WAV。請確認影片有音軌。",
    )


def load_pcm16_wav_as_tensor(wav_path: Path) -> torch.Tensor:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)

    if channels != 1 or sample_width != 2 or frame_rate != SAMPLE_RATE:
        raise RuntimeError("抽出的 WAV 格式不正確，應為單聲道 16-bit 16kHz。")
    if not frames:
        raise RuntimeError("抽出的 WAV 沒有音訊資料。")

    pcm = array.array("h")
    pcm.frombytes(frames)
    if sys.byteorder != "little":
        pcm.byteswap()

    return torch.tensor(pcm, dtype=torch.float32) / 32768.0


@st.cache_resource(show_spinner=False)
def get_vad_model():
    torch.set_num_threads(1)
    return load_silero_vad()


def detect_speech_segments(
    wav_path: Path,
    threshold: float,
    min_speech_ms: int,
) -> list[Segment]:
    wav = load_pcm16_wav_as_tensor(wav_path)
    model = get_vad_model()
    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=100,
        speech_pad_ms=0,
        return_seconds=True,
    )
    return [
        Segment(float(item["start"]), float(item["end"]))
        for item in timestamps
        if float(item["end"]) > float(item["start"])
    ]


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def merge_speech_by_silence(
    speech_segments: list[Segment],
    min_silence_seconds: float,
) -> list[Segment]:
    if not speech_segments:
        return []

    ordered = sorted(speech_segments, key=lambda item: item.start)
    merged = [ordered[0]]

    for segment in ordered[1:]:
        previous = merged[-1]
        gap = segment.start - previous.end
        if gap <= min_silence_seconds:
            merged[-1] = Segment(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)

    return merged


def build_keep_segments(
    speech_segments: list[Segment],
    video_duration: float,
    min_silence_seconds: float,
    padding_seconds: float,
) -> list[Segment]:
    merged_speech = merge_speech_by_silence(speech_segments, min_silence_seconds)
    if not merged_speech:
        return []

    keep_segments: list[Segment] = []
    for index, segment in enumerate(merged_speech):
        start = segment.start - padding_seconds
        end = segment.end + padding_seconds

        if index == 0 and segment.start <= min_silence_seconds:
            start = 0.0
        if index == len(merged_speech) - 1 and video_duration - segment.end <= min_silence_seconds:
            end = video_duration

        keep_segments.append(
            Segment(
                clamp(start, 0.0, video_duration),
                clamp(end, 0.0, video_duration),
            )
        )

    return merge_overlapping_segments(keep_segments)


def merge_overlapping_segments(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []

    ordered = sorted(segments, key=lambda item: item.start)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        previous = merged[-1]
        if segment.start <= previous.end:
            merged[-1] = Segment(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)

    return [segment for segment in merged if segment.duration > 0.01]


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, whole_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    if hours:
        return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return f"{minutes}:{whole_seconds:02d}.{milliseconds:03d}"


def ffmpeg_time(value: float) -> str:
    return f"{value:.6f}"


def build_concat_filter(segments: list[Segment]) -> str:
    lines: list[str] = []
    concat_inputs: list[str] = []

    for index, segment in enumerate(segments):
        start = ffmpeg_time(segment.start)
        end = ffmpeg_time(segment.end)
        lines.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]"
        )
        lines.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")

    lines.append(
        "".join(concat_inputs)
        + f"concat=n={len(segments)}:v=1:a=1[vcat][acat]"
    )
    lines.append("[vcat]format=yuv420p[vout]")
    return ";\n".join(lines)


def render_video(video_path: Path, output_path: Path, keep_segments: list[Segment]) -> None:
    if not keep_segments:
        raise RuntimeError("沒有可輸出的保留片段。")

    filter_graph = build_concat_filter(keep_segments)

    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[vout]",
            "-map",
            "[acat]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        "FFmpeg 輸出影片失敗。",
    )


def process_video(
    uploaded_file,
    min_silence_seconds: float,
    padding_seconds: float,
    threshold: float,
    min_speech_ms: int,
) -> dict[str, object]:
    require_ffmpeg()

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix.removeprefix(".") not in SUPPORTED_EXTENSIONS:
        raise RuntimeError("目前只支援 MP4、MOV、MKV。")

    with tempfile.TemporaryDirectory(prefix="vad_jumpcut_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        input_path = tmp_dir / f"input{suffix}"
        wav_path = tmp_dir / "audio_16k_mono.wav"
        output_path = tmp_dir / "auto_cut.mp4"

        input_path.write_bytes(uploaded_file.getbuffer())

        original_duration = get_video_duration(input_path)
        extract_mono_wav(input_path, wav_path)
        raw_speech_segments = detect_speech_segments(
            wav_path,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
        )
        keep_segments = build_keep_segments(
            raw_speech_segments,
            video_duration=original_duration,
            min_silence_seconds=min_silence_seconds,
            padding_seconds=padding_seconds,
        )

        if not keep_segments:
            raise RuntimeError(
                "Silero VAD 沒有偵測到可保留的語音區段。可以試著降低 VAD 靈敏度門檻，或確認音軌中有人聲。"
            )

        edited_duration = sum(segment.duration for segment in keep_segments)
        render_video(input_path, output_path, keep_segments)

        return {
            "video_bytes": output_path.read_bytes(),
            "original_duration": original_duration,
            "edited_duration": edited_duration,
            "deleted_duration": max(0.0, original_duration - edited_duration),
            "raw_speech_count": len(raw_speech_segments),
            "keep_count": len(keep_segments),
            "segments": keep_segments,
        }


def reset_previous_result_when_file_changes(uploaded_name: str | None) -> None:
    previous_name = st.session_state.get("uploaded_name")
    if previous_name != uploaded_name:
        st.session_state.pop("result", None)
        st.session_state["uploaded_name"] = uploaded_name


def main() -> None:
    st.set_page_config(
        page_title="AI 自動剪輯工具",
        layout="centered",
    )

    st.title("AI 自動剪輯工具")
    st.write("上傳影片後，系統會自動剪除超過門檻的無語音停頓，輸出成 MP4。")

    uploaded_file = st.file_uploader(
        "選擇影片",
        type=sorted(SUPPORTED_EXTENSIONS),
        accept_multiple_files=False,
    )
    reset_previous_result_when_file_changes(uploaded_file.name if uploaded_file else None)

    col_a, col_b = st.columns(2)
    with col_a:
        min_silence_seconds = st.slider(
            "最短停頓秒數",
            min_value=0.2,
            max_value=5.0,
            value=0.8,
            step=0.1,
            help="只有超過這個長度的無語音區段才會被剪掉。",
        )
    with col_b:
        padding_seconds = st.slider(
            "前後 padding 秒數",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.05,
            help="在每段語音前後多保留一點時間，避免尾音被切掉。",
        )

    with st.expander("進階設定"):
        threshold = st.slider(
            "VAD 語音判定門檻",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="數值越低越容易判定為有人聲；環境音很多時可調高。",
        )
        min_speech_ms = st.slider(
            "最短語音長度（毫秒）",
            min_value=50,
            max_value=1000,
            value=250,
            step=50,
            help="短於此長度的聲音片段會被忽略。",
        )

    if uploaded_file is not None:
        st.info(f"已選擇：{uploaded_file.name}")

    start = st.button(
        "開始自動剪輯",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    if start and uploaded_file is not None:
        progress = st.progress(0, text="準備處理影片")
        try:
            progress.progress(10, text="檢查 FFmpeg 與讀取影片")
            with st.spinner("正在分析語音並輸出影片，影片越長需要越多時間。"):
                result = process_video(
                    uploaded_file,
                    min_silence_seconds=min_silence_seconds,
                    padding_seconds=padding_seconds,
                    threshold=threshold,
                    min_speech_ms=min_speech_ms,
                )
            progress.progress(100, text="完成")
            st.session_state["result"] = result
        except Exception as exc:
            progress.empty()
            st.error(str(exc))

    result = st.session_state.get("result")
    if result:
        st.success("剪輯完成")

        original_duration = float(result["original_duration"])
        edited_duration = float(result["edited_duration"])
        deleted_duration = float(result["deleted_duration"])

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("原始長度", format_seconds(original_duration))
        metric_b.metric("剪輯後長度", format_seconds(edited_duration))
        metric_c.metric("刪除秒數", format_seconds(deleted_duration))

        st.caption(
            f"偵測到 {result['raw_speech_count']} 個語音片段，輸出時合併為 {result['keep_count']} 個保留片段。"
        )
        st.video(result["video_bytes"], format="video/mp4")
        st.download_button(
            "下載剪輯後 MP4",
            data=result["video_bytes"],
            file_name="auto_cut.mp4",
            mime="video/mp4",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
