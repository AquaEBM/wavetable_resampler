fn empty_buffers_with_capacity(capacity: usize, buffer_count: usize) -> Vec<Vec<f32>> {
    use std::iter;
    iter::repeat_with(|| Vec::with_capacity(capacity)).take(buffer_count).collect()
}

fn resample_into(input: &[Vec<f32>], output: &mut Vec<Vec<f32>>, xs: &[f32], ys: &[f32]) {

    use plugin_util::dsp::{lerp_table, lerp_terrain};

    for (&y, buffer) in ys.iter().zip(output.iter_mut()) {
        for &x in xs.iter() {
            buffer.push( unsafe { lerp_terrain(input, x, y) } );
        }
    }
    let last_out = output.last_mut().unwrap();
    let last_in = input.last().unwrap();
    for &x in xs.iter() {
        last_out.push( unsafe { lerp_table(last_in, x) } );
    }
}

use std::path::Path;

/// assumes each buffer in `input_buffers` is empty
pub fn read_i16_wave_frame(path: impl AsRef<Path>, input_buffers: &mut Vec<Vec<f32>>) {

    use hound::WavReader;

    let reader = WavReader::open(path).unwrap();
    const MAX: f32 = i16::MAX as f32;

    let mut samples = reader.into_samples::<i16>();
    let samples_iter = samples.by_ref();

    let window_size = input_buffers[0].capacity() - 1;

    input_buffers.iter_mut().for_each( |buffer| {
        buffer.extend(samples_iter.take(window_size).map(|res| res.unwrap() as f32 / MAX));
        buffer.push(*buffer.first().unwrap());
    });
}

pub fn resample_file(
    path: impl AsRef<Path>,
    input_buffers: &mut Vec<Vec<f32>>,
    output_buffers: &mut Vec<Vec<f32>>,
    xs: &[f32],
    ys: &[f32],
) {
    read_i16_wave_frame(path.as_ref(), input_buffers);
    resample_into(input_buffers, output_buffers, xs, ys);

    use hound::{SampleFormat, WavSpec, WavWriter};

    const WAV_WRITER_SPEC: WavSpec = WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    use std::{fs::File, io::BufWriter};

    // 2MB is about the size of a 2048 x 256 samples 32 bit FP wav file
    let file = BufWriter::with_capacity(2_102_000, File::create(path.as_ref()).unwrap());
    let mut writer = WavWriter::new(file, WAV_WRITER_SPEC).unwrap();

    output_buffers.iter()
        .map(|slice| plugin_util::all_but_last(slice))
        .flatten()
        .for_each(|&sample| writer.write_sample(sample).unwrap());
}

fn main() {

    use std::{env, fs::read_dir};

    let mut args = env::args().skip(1);
    let args_iter = args.by_ref();

    let mut parse_next_arg = || args_iter.next().unwrap().parse().unwrap();

    let input_window_len = parse_next_arg();
    let input_frame_count = parse_next_arg();
    let output_window_len = parse_next_arg();
    let output_frame_count = parse_next_arg();
    let path = args_iter.next().unwrap();

    let mut input_buffers = empty_buffers_with_capacity(input_window_len + 1, input_frame_count);
    let mut output_buffers = empty_buffers_with_capacity(output_window_len, output_frame_count);

    let num_frames_ratio = (input_frame_count - 1) as f32 / (output_frame_count - 1) as f32;
    let window_size_ratio = input_window_len as f32 / output_window_len as f32;

    let xs: Vec<_> = (0..output_window_len).map(|i| i as f32 * window_size_ratio).collect();
    let ys: Vec<_> = (0..output_frame_count - 1).map(|i| i as f32 * num_frames_ratio).collect();

    for path in read_dir(path).unwrap() {
        resample_file(path.unwrap().path(), &mut input_buffers, &mut output_buffers, &xs, &ys);
        input_buffers.iter_mut().for_each(Vec::clear);
        output_buffers.iter_mut().for_each(Vec::clear);
    }
}