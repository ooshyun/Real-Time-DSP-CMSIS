# ffmpeg -f s16le -ar 48k -ac 1 -i ./data/221026_tmp/voice_0_202210261320_ap41dbfs_dspgain_0_0.pcm result.wav

# wav to pcm
# ffmpeg -y -i ./WhiteNoise_Full_-40_dBFS_48k_PCM16_LR.wav -acodec pcm_s16le -f s16le -ac 2 -ar 48000 output.pcm

# pcm to wav
# ffmpeg -f s16le -ar 48k -ac 2 -i ./WhiteNoise_Full_-40_dBFS_48k_PCM16_LR.pcm result.wav



# wav to pcm
ffmpeg -y -ac 1 -i -ar 44100 ./test/data/sin_1k_tone.wav -acodec pcm_s16le -f s16le -ac 1 -ar 44100 ./test/data/sin_1k_tone.pcm


# pcm to wav
ffmpeg -f s16le -ar 44100 -ac 1 -i ./WhiteNoise_Full_-40_dBFS_48k_PCM16_LR.pcm result.wav

ffmpeg -f s16le -ar 44100 -ac 1 -i ./WhiteNoise_Full_-40_dBFS_48k_PCM16_LR.pcm result.wav

ffmpeg -f s16le -ar 44100 -ac 1 -i ./test/data/sin_1k_stereo_tone.pcm ./test/data/result_stereo.wav

ffmpeg -f s16le -ar 44100 -ac 2 -i ./test/data/sin_1k_stereo_tone.pcm ./test/data/result_stereo.wav