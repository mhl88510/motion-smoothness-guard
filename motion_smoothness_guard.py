#!/usr/bin/env python3
"""Screen video for repeated and near-static frames; does not certify quality."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

DEFAULTS = {
    'near_static_diff_threshold': 0.5,
    'max_near_static_ratio': 0.40,
    'max_consecutive_near_static_pairs': 15,
    'max_consecutive_exact_duplicates': 2,
    'analysis_width': 96,
    'analysis_height': 170,
}


def ffprobe(path: Path) -> dict:
    result = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'format=duration:stream=codec_type,r_frame_rate,avg_frame_rate,nb_frames',
        '-of', 'json', str(path),
    ], text=True)
    return json.loads(result)


def json_input(value: str):
    if value.lstrip().startswith(('[', '{')):
        return json.loads(value)
    return json.loads(Path(value).read_text(encoding='utf-8-sig'))


def parse_ranges(value: str | None) -> list[tuple[float, float]]:
    if not value:
        return []
    data = json_input(value)
    if not isinstance(data, list):
        raise ValueError('Static ranges must be a JSON array')
    result = []
    for pair in data:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('Each static range needs a start and end')
        if any(isinstance(x, bool) for x in pair):
            raise ValueError('Range bounds must be numbers')
        start, end = map(float, pair)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError('Static ranges require finite 0 <= start < end')
        result.append((start, end))
    return result


def parse_cuts(value: str | None) -> list[float]:
    if not value:
        return []
    if value.lstrip().startswith(('[', '{')) or Path(value).is_file():
        data = json_input(value)
        if isinstance(data, dict):
            data = data.get('cuts', data.get('cut_times_seconds', []))
    else:
        data = value.split(',')
    if not isinstance(data, list) or any(isinstance(x, bool) for x in data):
        raise ValueError('Cuts must be an array or comma-separated numbers')
    cuts = [float(x) for x in data]
    if any(not math.isfinite(x) or x < 0 for x in cuts):
        raise ValueError('Cuts must be finite non-negative seconds')
    return sorted(set(cuts))


def max_run(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def decide(metrics: dict, cfg: dict | None = None) -> tuple[str, list[str]]:
    config = {**DEFAULTS, **(cfg or {})}
    reasons = []
    if not metrics['cfr_pass']:
        reasons.append('VFR_OR_FRAME_COUNT_MISMATCH')
    if metrics['near_static_ratio'] > config['max_near_static_ratio']:
        reasons.append('NEAR_STATIC_RATIO_EXCEEDED')
    if metrics['max_consecutive_near_static_pairs'] > config['max_consecutive_near_static_pairs']:
        reasons.append('NEAR_STATIC_RUN_EXCEEDED')
    if metrics['max_consecutive_exact_duplicate_frames'] > config['max_consecutive_exact_duplicates']:
        reasons.append('EXACT_DUPLICATE_RUN_EXCEEDED')
    if metrics.get('analyzed_pairs') == 0:
        reasons.append('NO_ANALYZABLE_FRAME_PAIRS')
    return ('RETURN' if reasons else 'PASS', reasons)


def measure_frames(frames, fps: float, cuts: list[float], allow_static, config: dict) -> dict:
    """Measure an iterator one frame at a time; exclusions break consecutive runs."""
    cut_frames = {round(t * fps) for t in cuts}
    count = analyzed = excluded = near_count = exact_count = 0
    near_run = exact_run = longest_near = longest_exact = 0
    difference_sum = 0.0
    previous = None
    for frame in frames:
        index = count
        count += 1
        if previous is not None:
            is_cut = index in cut_frames or index - 1 in cut_frames
            intentional = any(a <= index / fps < b for a, b in allow_static)
            if is_cut or intentional:
                excluded += 1
                near_run = exact_run = 0
            else:
                difference = float(np.mean(np.abs(frame.astype(np.int16) - previous.astype(np.int16))))
                exact = bool(np.array_equal(frame, previous))
                near = difference < config['near_static_diff_threshold']
                analyzed += 1
                difference_sum += difference
                near_count += near
                exact_count += exact
                near_run = near_run + 1 if near else 0
                exact_run = exact_run + 1 if exact else 0
                longest_near = max(longest_near, near_run)
                longest_exact = max(longest_exact, exact_run)
        previous = frame
    return {
        'frames': count, 'fps': fps, 'analyzed_pairs': analyzed, 'excluded_pairs': excluded,
        'mean_abs_diff': round(difference_sum / analyzed, 4) if analyzed else None,
        'near_static_pairs': near_count, 'near_static_ratio': round(near_count / analyzed, 6) if analyzed else 0.0,
        'max_consecutive_near_static_pairs': longest_near, 'exact_duplicate_pairs': exact_count,
        # This historical field counts equal adjacent PAIRS, not total frames in a run.
        'max_consecutive_exact_duplicate_frames': longest_exact,
    }


def decoded_frames(path: Path, config: dict):
    width, height = config['analysis_width'], config['analysis_height']
    frame_size = width * height
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([
            'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', str(path),
            '-map', '0:v:0', '-vf', f'scale={width}:{height}:flags=bilinear,format=gray',
            '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray', '-',
        ], stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                chunks = bytearray()
                while len(chunks) < frame_size:
                    part = process.stdout.read(frame_size - len(chunks))
                    if not part:
                        break
                    chunks.extend(part)
                if not chunks:
                    break
                if len(chunks) != frame_size:
                    raise ValueError('Decoder returned an incomplete frame')
                yield np.frombuffer(chunks, dtype=np.uint8).reshape(height, width)
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError('FFmpeg decoding failed: ' + errors.read(2048).decode('utf8', errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def analyze(video: Path, cuts: list[float], allow_static, cfg: dict | None = None) -> dict:
    config = {**DEFAULTS, **(cfg or {})}
    probe = ffprobe(video)
    streams = [s for s in probe.get('streams', []) if s.get('codec_type') == 'video']
    if not streams:
        raise ValueError('Input contains no video stream')
    stream = streams[0]
    rate, average = Fraction(stream['r_frame_rate']), Fraction(stream['avg_frame_rate'])
    fps = float(average)
    duration = float(probe['format']['duration'])
    if not (math.isfinite(fps) and fps > 0 and math.isfinite(duration) and duration > 0):
        raise ValueError('Video needs positive, finite frame rate and duration')
    declared_value = stream.get('nb_frames')
    declared = int(declared_value) if declared_value not in (None, 'N/A', '') else 0
    frames = decoded_frames(video, config)
    try:
        metrics = measure_frames(frames, fps, cuts, allow_static, config)
    finally:
        frames.close()
    metrics.update(duration_seconds=duration, r_frame_rate=str(rate), avg_frame_rate=str(average), declared_nb_frames=declared)
    metrics['cfr_pass'] = (rate == average and (not declared or abs(declared - metrics['frames']) <= 1)
                           and abs(metrics['frames'] - duration * fps) <= 2)
    status, reasons = decide(metrics, config)
    return {'schema': 'motion-smoothness-guard.v1', 'status': status, 'reasons': reasons,
            'video': str(video.resolve()), 'video_sha256': digest(video), 'thresholds': config,
            'cuts_seconds': cuts, 'declared_static_ranges_seconds': allow_static, 'metrics': metrics,
            'interpretation': 'Heuristic frame repetition screen. Frame-rate metadata consistency is not a per-frame timestamp proof. PASS does not certify aesthetics, music fit, or human acceptance.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--cuts')
    parser.add_argument('--allow-static')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.out.resolve() == args.video.resolve():
            raise ValueError('Report cannot overwrite the input video')
        result = analyze(args.video, parse_cuts(args.cuts), parse_ranges(args.allow_static))
    except (OSError, ValueError, KeyError, ZeroDivisionError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Error: {error}\n')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'reasons': result['reasons'], 'metrics': result['metrics']}))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
