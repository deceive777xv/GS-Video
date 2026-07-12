# Media fixtures

`source.mp4` is a 10-second, 320x180, 10 fps H.264 video with a 440 Hz AAC audio track.
It contains exactly 100 video frames and was generated with FFmpeg 8.1 using:

```powershell
ffmpeg -y -f lavfi -i "testsrc2=size=320x180:rate=10" -f lavfi -i "sine=frequency=440" -t 10 -c:v libx264 -pix_fmt yuv420p -c:a aac tests/fixtures/media/source.mp4
```

- Size: 264,237 bytes
- SHA-256: `A92D867374D78EBC696B689C4C07A5D4E6EC585006F88BCFD8BCE4F32C426FDF`
