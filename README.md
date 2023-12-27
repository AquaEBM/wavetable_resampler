# Wavetable Resampler

Little program that resamples wavetables (represented as WAV files of size <samples_per_frame> x <number_of_frames>) to another size using FFT to resample each frame and linear interpolation to crossfade between them

Mainly used to adapt the wavetables from [WaveEdit](https://waveeditonline.com/) (256x64) to my synthesizer [Jade](https://github.com/AquaEBM/Jade) which requires 2048x256 wavetables.
