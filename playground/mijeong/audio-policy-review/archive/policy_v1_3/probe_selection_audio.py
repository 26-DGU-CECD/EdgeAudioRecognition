"""Read selected WAV files and verify provenance without ML/audio framework packages."""
import argparse
import struct
from pathlib import Path

import numpy as np

from project_io import write_json
from selection_replay_io import load_bundle, verify_audio, verify_checkpoint


def read_excerpt(path):
    with Path(path).open('rb') as stream:
        signature = stream.read(12)
        if signature[:4] != b'RIFF' or signature[8:12] != b'WAVE':
            raise ValueError('Expected a little-endian RIFF/WAVE file')
        fmt = None
        while True:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError('No audio data chunk found')
            name, size = struct.unpack('<4sI', chunk)
            if name == b'fmt ':
                fmt = stream.read(size)
                if size % 2:
                    stream.seek(1, 1)
            elif name == b'data':
                if fmt is None or len(fmt) < 16:
                    raise ValueError('Missing WAV format header')
                code, channels, rate, _, align, bits = struct.unpack('<HHIIHH', fmt[:16])
                if code == 0xFFFE and len(fmt) >= 40:
                    code = struct.unpack('<H', fmt[24:26])[0]
                dtype = {(3, 32): '<f4', (3, 64): '<f8', (1, 16): '<i2', (1, 32): '<i4'}.get((code, bits))
                if not dtype or channels < 1 or rate < 1 or align < 1 or size % align:
                    raise ValueError('Unsupported or invalid WAV sample layout; check with SoundFile')
                if align != channels * np.dtype(dtype).itemsize:
                    raise ValueError('WAV block alignment differs from sample format')
                count = min(size, round(2 * rate) * align)
                raw = stream.read(count)
                if len(raw) != count:
                    raise ValueError('Truncated audio samples')
                samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
                if code == 1:
                    samples /= 2 ** (bits - 1)
                if not len(samples) or not np.isfinite(samples).all():
                    raise ValueError('Empty or non-finite audio')
                return dict(sample_rate=rate, channels=channels, format_code=code,
                            duration_seconds=size / align / rate, sampled_frames=len(samples) // channels,
                            samples_all_finite=True, sampled_abs_peak=float(np.abs(samples).max()))
            else:
                stream.seek(size + size % 2, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    _, manifest, fingerprint = load_bundle(args.out)
    for model in manifest['checkpoint_files']:
        verify_checkpoint(manifest, model)
    results = []
    for episode in manifest['episodes']:
        path = verify_audio(episode)
        result = dict(label=episode['label'], path=str(path), **read_excerpt(path))
        results.append(result)
        print(episode['label'], result['sample_rate'], 'Hz', result['channels'], 'channel(s)',
              result['duration_seconds'], 's', 'finite samples: OK')
    write_json(Path(args.out) / 'selection_audio_probe.json', dict(
        fingerprint=fingerprint, episodes_checked=len(results), checkpoint_hashes_checked=True,
        files=results, scope='Source file access, WAV header and first 2s samples only; no model or preprocessing execution.',
    ))
    print('Saved selection_audio_probe.json')


if __name__ == '__main__':
    main()
