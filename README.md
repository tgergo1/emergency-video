# Emergency Video Safer-Floor Codec Demo (C++ + OpenCV)

This project captures live webcam video with OpenCV, then uses a custom C++ codec (standard library only for compression logic) to encode/decode a very low-bitrate grayscale stream focused on scene recognizability.
The codec engine runs in-process, while a built-in web UI provides camera selection, controls, and live visualization.

## Features

- Webcam capture with runtime camera selection
- Browser dashboard (no OpenCV text overlays on frames)
- Role-based dashboard tabs: `Link`, `Send`, `Receive`, `Status`, `Advanced`
- Triple live feed dashboard:
  - `RAW INPUT`
  - `SENT (DITHERED)`
  - `RECEIVED (INTERP+ENH)`
- Internal codec profile tuned for unknown emergency-scene content
- Grayscale-only 4-bit luma internal coding
- Periodic keyframes + interframe temporal prediction
- Block-level sparse updates with motion-saliency capping
- Face-aware ROI updates (detected faces are prioritized for inter updates)
- Motion-compensated interpolation on the receiver path for smoother low-fps playback
- Receiver-side deblock + edge-aware enhancement
- Acoustic link transport with custom packet framing:
  - `local_loopback`
  - `acoustic_tx`
  - `acoustic_rx_live`
  - `acoustic_rx_media`
  - `acoustic_duplex_arq` (half-duplex TDMA ARQ)
- Miniaudio capture/playback path for live mic/speaker transport
- Media-file receive path (demod from decoded audio track)
- Config beacons + config hash/version checks before payload decode
- Per-frame CRC + repetition/interleave FEC
- ACK/retransmit path in duplex ARQ mode
- Custom bitstream with custom bit writer/reader
- Live metrics: bytes/frame, live/smoothed/average bitrate, compression ratios, encoded fps, keyframe ratio, changed blocks, PSNR
- Live bandwidth bar visualization
- Web controls are fully wired (no inert fields): transport, link role, serial settings, relay paths, text send, codec/display toggles, and link start/stop

## Build

Requirements:

- CMake >= 3.16
- C++17 compiler
- OpenCV modules: `core`, `imgproc`, `highgui`, `videoio`, `objdetect`
- `pkg-config`
- FFmpeg development libraries: `libavformat`, `libavcodec`, `libavutil`, `libswresample`

Build steps:

```bash
cmake -S . -B build
cmake --build build -j
```

Portable package (same CMake workflow intended for macOS/Windows/Linux):

```bash
cmake --build build --target portable
```

This creates a self-contained folder at:

```text
build/portable
```

Run from there:

```bash
./build/portable/emergency_video
```

Portable notes:

- This is a **portable folder**, not a single self-contained binary.
- Move the whole `build/portable` directory to another machine with the same OS/arch.
- The packaging flow is the same across macOS/Linux/Windows (`cmake --build build --target portable`), while copied runtime library types differ by platform (`.dylib` / `.so` / `.dll`).
- For the most aggressive dependency bundling, configure with:
  - `-DEV_PORTABLE_BUNDLE_SYSTEM_LIBS=ON`

If OpenCV is not installed locally, CMake will automatically fetch/build OpenCV (tag `4.10.0`) using `FetchContent`.
This can make the first configure/build significantly longer.

System OpenCV only (disable fetch fallback):

```bash
cmake -S . -B build -DEV_FETCH_OPENCV=OFF -DOpenCV_DIR=/path/to/OpenCVConfig.cmake
cmake --build build -j
```

Run:

```bash
./build/emergency_video
```

Then open:

```text
http://localhost:8080
```

Optional legacy OpenCV window UI build:

```bash
cmake -S . -B build -DEV_BUILD_LEGACY_OPENCV_UI=ON
cmake --build build -j --target emergency_video_legacy
./build/emergency_video_legacy
```

Run tests locally:

```bash
cmake -S . -B build -DEV_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure --verbose
```

Unit-test coverage includes:

- bitstream read/write round-trip and header round-trip
- codec encode/decode round-trip in safer and aggressive modes
- corruption handling (invalid sync, truncation, trailing garbage)
- decoder state handling (interframe without reference, reset behavior)
- malformed change-map rejection
- keyframe interval behavior and frame-index progression
- deterministic encoding checks (same input/state -> same bytes)
- communicator envelope fragmentation/reassembly checks
- communicator envelope corruption rejection and queue dedup checks

## CI Artifacts

GitHub Actions workflow:

- `.github/workflows/portable-artifacts.yml`

What it does on each push/PR/manual run:

- Builds on `ubuntu-latest`, `macos-latest`, and `windows-latest` (MSYS2).
- Runs CTest codec/unit tests.
- Builds the `portable` package.
- Validates packaged runtime dependencies (`ldd`/`otool` checks).
- Uploads downloadable portable artifacts per platform.

## Controls

- `Link` tab: essential setup for non-technical use (alias, camera, transport, role, quality, resolution, fps, start/stop).
- `Send` tab: reliable text channel + quick emergency phrases + timeline.
- `Receive` tab: relay import/export actions.
- `Status` tab: queue/link/codec diagnostics.
- `Advanced` tab: acoustic/serial specifics and codec/display toggles.
- Transport role selector uses `send` / `receive` / `duplex`.
- Start camera preference can be set with `EV_CAMERA_INDEX` (default `1`)

HTTP API (v2 only):

- `GET /api/v2/state`
- `POST /api/v2/control`
- `POST /api/v2/link/start`
- `POST /api/v2/link/stop`
- `POST /api/v2/messages/send`
- `GET /api/v2/frame/{raw|sent|received}.jpg`

Face detector note:

- Set `EV_FACE_CASCADE` to your Haar cascade path if auto-discovery fails.
- Typical filename: `haarcascade_frontalface_default.xml`.

Camera note:

- The app now explicitly prefers camera index `1` by default.
- If your internal camera is on another index, run for example:
  - `EV_CAMERA_INDEX=1 ./build/emergency_video`

## Codec Design

### Pipeline

1. Capture full-color webcam frame via OpenCV.
2. Resize to internal encoded resolution.
3. Convert to grayscale.
4. Quantize each pixel to 4-bit luma (`0..15`).
5. Encode frame as keyframe or interframe.
6. Decode immediately from produced bitstream.
7. Optional enhancement/dithering for display.

### Profiles

- `safer` (default): `128x96`, target `~2.5 fps`, keyframe interval `12`
- `aggressive`: `96x72`, target `~2.0 fps`, keyframe interval `18`

### Block Coding

- Block size: `8x8`
- Keyframe blocks:
  - absolute `4x4` cell representation (16 values, 4 bits each), or
  - raw `8x8` 4-bit samples for difficult blocks
- Interframe blocks:
  - unchanged blocks skipped via change map
  - changed blocks use one of:
    - residual `4x4` cells vs previous reconstructed frame
    - absolute `4x4` cells
    - raw `8x8` 4-bit samples

The encoder always predicts from the **previous reconstructed frame**.

### Sparse Change Strategy

- Block change detection uses MAE and max-difference thresholds.
- Candidate changed blocks are ranked by motion saliency score.
- A per-profile cap (`maxChangedFraction`) keeps only the strongest changed blocks.
- Non-selected blocks are copied from previous reconstruction.

This acts as a low-bitrate safety floor when scene motion spikes.

### Bitstream Layout

Each frame bitstream includes:

- frame sync + version
- frame type (key/inter)
- codec mode
- dimensions and block size
- residual step
- frame index
- keyframe interval
- total block count
- changed block count
- payload:
  - keyframe block payloads, or
  - interframe RLE change map + changed block payloads

Change map uses run-length coding over block states (`unchanged`/`changed`), starting from `unchanged`.

## Why This Fits Unknown Emergency Content

- Always keeps global context through periodic keyframes.
- Preserves coarse scene structure with low-resolution cell coding.
- Prioritizes moving/high-change regions under bitrate pressure.
- Core codec remains deterministic and block-based; optional face-ROI prioritization is a lightweight assist.
- Uses deterministic, transparent logic suitable for on-device fallback pipelines.

## How Bitrate Is Reduced

- Lower internal resolution (`128x96` or `96x72`)
- Grayscale-only coding
- 4-bit luma quantization
- Low encoded fps (`~2-3`)
- Sparse interframe updates (skip unchanged blocks)
- Saliency-based cap on changed block count
- Coarse `4x4` block payload modes instead of always raw `8x8`
- Run-length coding for block change map

## How to Tune for Lower Bitrate

- Switch to `aggressive` mode in the UI.
- Increase keyframe interval (edit `makeCodecParams` values).
- Raise `skipAvgThreshold` / `skipMaxThreshold`.
- Lower `maxChangedFraction`.
- Increase `residualStep`.
- Lower `targetFps`.
- Use `96x72` or lower encoded resolution.

## How to Tune for Better Recognizability

- Use `safer` mode.
- Reduce keyframe interval (use `default` instead of `short`).
- Lower skip thresholds to update more blocks.
- Increase `maxChangedFraction`.
- Lower `residualStep`.
- Keep receiver enhancement enabled.
- Keep dithering enabled if banding is distracting.

## Limitations

- Not a standards-compliant media codec.
- Acoustic transport is best-effort and environment dependent (speaker/mic quality, room echo, noise, AGC).
- Current implementation uses simple repetition/interleave FEC rather than full Reed-Solomon.
- FFmpeg is linked as a library dependency at build/runtime (single-binary static packaging is not completed here).
- No global motion compensation or advanced entropy coding.
- Severe motion or camera shake can still force visible staleness or blocking.
- PSNR is computed in the internal 4-bit domain (useful for trend tracking, not perceptual truth).
