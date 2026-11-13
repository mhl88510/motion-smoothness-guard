# Motion Smoothness Guard

A Python CLI that screens video for duplicate-frame runs and near-static sections. It produces a JSON report tied to the video's SHA-256 hash and lets you exclude intentional static sections and declared hard cuts.

这是视频交付前的辅助检查工具：检测重复帧与近静止段，输出可追溯报告。通过不等于画面好看，也不能替代人工连续播放检查。

## Requirements and usage

Python 3.10+, NumPy, and FFmpeg 5+ including `ffmpeg` and `ffprobe` on `PATH`. Install FFmpeg from your OS package manager or the builds linked by https://ffmpeg.org/download.html . No account, API key, or network access is used by this tool.

```sh
python -m pip install -r requirements.txt
python motion_smoothness_guard.py --video clip.mp4 --out output/report.json
python motion_smoothness_guard.py --video clip.mp4 --cuts "1.2,3.5" --allow-static "[[0,0.5]]" --out output/report.json
python -m unittest discover -v
```

`--cuts` accepts seconds separated by commas, a JSON array, or a JSON file with `cuts` / `cut_times_seconds`. `--allow-static` accepts a JSON array of `[start,end]` seconds or a file containing that array.

Exit codes: **0** for heuristic PASS, **2** for RETURN (review recommended), **1** for input/decode errors. Existing report files are replaced; the input video cannot be used as the report path.

## What the numbers mean

- Frames are streamed through a 96×170 grayscale analysis raster. Only the current and preceding frame are retained by the metric calculation.
- Mean absolute luma difference below 0.5 is considered near-static. The defaults flag a near-static ratio above 40%, more than 15 consecutive near-static pairs, or more than two equal adjacent pairs.
- Hard cuts exclude the transition pair and its following pair. Intentional-static ranges exclude pairs by the current frame time. An excluded pair resets consecutive-run counters.
- The historical field `max_consecutive_exact_duplicate_frames` counts equal adjacent **pairs**: three identical frames produce a value of two.
- All pairs excluded, a single frame, or no analyzable pairs result in RETURN, not PASS.
- Frame-rate metadata and decoded count consistency are checked. This is a heuristic, not proof of constant spacing of all presentation timestamps. Variable-rate time ranges are estimated using the average frame rate.
- Thresholds originated in short vertical montage work. Landscape footage, slides, interviews, animation and other styles may need different thresholds. Low movement can be intentional.
- Reports contain the input's local absolute path; review them before sharing.

## Origin, tests and license

Generalized from the author's private video-production tools. Excluded-pair run counting, bounded-memory decoding and input validation have been improved for this standalone release. Examples/tests use synthetic frames rather than production media or client data.

Unit tests validate metrics, exclusions, parser errors and report decisions. The synthetic-video integration test runs only when both FFmpeg executables are available. No actual production film is bundled or certified by these tests.

MIT license; see [LICENSE](LICENSE).
