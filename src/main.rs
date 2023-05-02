use std::{io::{BufReader, BufWriter}, fs::{File, read_dir}, env, sync::Arc};
use hound::{SampleFormat, WavSpec, WavWriter, WavReader};
use realfft::{RealFftPlanner, num_complex::Complex32, RealToComplex, ComplexToReal, num_traits::Zero};

struct WaveTable {
    buffer: Vec<f32>,
    frame_len: usize,
    frame_count: usize,
}

impl WaveTable {

    fn new(frame_len: usize, frame_count: usize) -> Self {

        let size = frame_len * frame_count;
    
        let mut buffer = Vec::with_capacity(size);
    
        unsafe { buffer.set_len(size) }

        Self {
            buffer,
            frame_len,
            frame_count
        }
    }

    fn read(&mut self, reader: WavReader<&mut BufReader<File>>) {

        assert!(self.frame_len * self.frame_count == reader.len() as usize);

        let samples = reader.into_samples::<i16>().map(Result::unwrap);

        self.buffer
            .iter_mut()
            .zip(samples)
            .for_each(|(output, input)| {
                const MAX: f32 = i16::MAX as f32;
                *output = input as f32 / MAX
            });
    }

    fn resample_into(&self, wt: &mut WaveTable, fft_data: &mut ResamplingCache) {

        let input_inter_frame_num_harmonics = self.frame_count / 2 + 1;
        let output_inter_frame_num_harmonics = wt.frame_count / 2 + 1;

        for i in 0..self.frame_len {
            self.buffer[i..]
                .iter()
                .step_by(self.frame_len)
                .zip(fft_data.inter_frame_scratch_input.iter_mut())
                .for_each(|(&sample, output)| *output = sample);

            fft_data.inter_frame_ifft.process_with_scratch(
                &mut fft_data.inter_frame_scratch_input,
                &mut fft_data.general_spectrum_output[..input_inter_frame_num_harmonics],
                &mut fft_data.general_complex_scratch
            ).unwrap();

            fft_data.general_spectrum_output[
                input_inter_frame_num_harmonics..output_inter_frame_num_harmonics
            ].fill(Complex32::zero());

            fft_data.inter_frame_fft.process_with_scratch(
                &mut fft_data.general_spectrum_output[..output_inter_frame_num_harmonics],
                &mut fft_data.inter_frame_scratch_output,
                &mut fft_data.general_complex_scratch
            ).unwrap();

            let norm = self.frame_count as f32;

            fft_data.inter_frame_scratch_output.iter_mut().for_each(|sample| *sample /= norm);

            fft_data.inter_frame_buffer[i..]
                .iter_mut()
                .step_by(self.frame_len)
                .zip(fft_data.inter_frame_scratch_output.iter())
                .for_each(|(output, &sample)| *output = sample);
        }

        let input_frame_num_harmonics = self.frame_len / 2 + 1;
        let output_frame_num_harmonics = wt.frame_len / 2 + 1;

        for (output, input) in wt.buffer
            .chunks_exact_mut(wt.frame_len)
            .zip(fft_data.inter_frame_buffer.chunks_exact_mut(self.frame_len))
        {
            fft_data.per_frame_ifft.process_with_scratch(
                input,
                &mut fft_data.general_spectrum_output[..input_frame_num_harmonics],
                &mut fft_data.general_complex_scratch
            ).unwrap();

            fft_data.general_spectrum_output[
                input_frame_num_harmonics..output_frame_num_harmonics
            ].fill(Complex32::zero());

            fft_data.general_spectrum_output[0] = Complex32::zero();

            fft_data.per_frame_fft.process_with_scratch(
                &mut fft_data.general_spectrum_output[..output_frame_num_harmonics],
                output,
                &mut fft_data.general_complex_scratch
            ).unwrap();

            let norm = self.frame_len as f32;

            output.iter_mut().for_each(|sample| *sample /= norm);
        }
    }

    fn write(&self, mut writer: WavWriter<&mut BufWriter<File>>) {

        self.buffer.iter().for_each(|&sample| writer.write_sample(sample).unwrap());

        writer.finalize().unwrap();
    }
}

struct ResamplingCache {
    general_spectrum_output: Vec<Complex32>,
    general_complex_scratch: Vec<Complex32>,
    inter_frame_buffer: Vec<f32>,
    inter_frame_scratch_input: Vec<f32>,
    inter_frame_scratch_output: Vec<f32>,
    inter_frame_ifft: Arc<dyn RealToComplex<f32>>,
    inter_frame_fft: Arc<dyn ComplexToReal<f32>>,
    per_frame_ifft: Arc<dyn RealToComplex<f32>>,
    per_frame_fft: Arc<dyn ComplexToReal<f32>>,
}

impl ResamplingCache {
    fn new(
        input_frame_len: usize,
        input_frame_count: usize,
        output_frame_len: usize,
        output_frame_count: usize
    ) -> Self {

        let mut fft = RealFftPlanner::<f32>::new();

        let intermediate_buffer_len = input_frame_len * output_frame_count;

        let greatest_value = *[input_frame_len, input_frame_count, output_frame_len, output_frame_count]
            .iter()
            .max()
            .unwrap();

        let mut inter_frame_buffer = Vec::with_capacity(intermediate_buffer_len);
        let mut general_complex_scratch = Vec::with_capacity(greatest_value);
        let mut inter_frame_scratch_input = Vec::with_capacity(input_frame_count);
        let mut inter_frame_scratch_output = Vec::with_capacity(output_frame_count);

        unsafe {
            inter_frame_buffer.set_len(intermediate_buffer_len);
            general_complex_scratch.set_len(greatest_value);
            inter_frame_scratch_input.set_len(input_frame_count);
            inter_frame_scratch_output.set_len(output_frame_count);
        }

        let inter_frame_ifft = fft.plan_fft_forward(input_frame_count);
        let inter_frame_fft = fft.plan_fft_inverse(output_frame_count);

        let per_frame_ifft = fft.plan_fft_forward(input_frame_len);
        let per_frame_fft = fft.plan_fft_inverse(output_frame_len);

        Self {
            general_spectrum_output: general_complex_scratch.clone(),
            general_complex_scratch,
            inter_frame_buffer,
            inter_frame_scratch_input,
            inter_frame_scratch_output,
            inter_frame_ifft,
            inter_frame_fft,
            per_frame_ifft,
            per_frame_fft
        }
    }
}

const SPEC: WavSpec = WavSpec {
    bits_per_sample: 32,
    channels: 1,
    sample_rate: 44100,
    sample_format: SampleFormat::Float
};

fn main() {

    let mut args = env::args().skip(1);
    let args_iter = args.by_ref();

    let mut parse_next_arg = || args_iter.next().unwrap().parse::<usize>().unwrap();

    let input_frame_len = parse_next_arg();
    let input_frame_count = parse_next_arg();
    let output_frame_len = parse_next_arg();
    let output_frame_count = parse_next_arg();
    let path = args_iter.next().unwrap();

    let mut input_wt = WaveTable::new(input_frame_len, input_frame_count);
    let mut output_wt = WaveTable::new(output_frame_len, output_frame_count);

    let mut paths = read_dir(path).unwrap().map(Result::unwrap);

    let first_entry = paths.next().unwrap();

    let first_path = first_entry.path();

    let mut fft_data = ResamplingCache::new(
        input_frame_len,
        input_frame_count,
        output_frame_len,
        output_frame_count,
    );

    let mut reader = BufReader::with_capacity(
        input_frame_len * input_frame_count * 2 + 1024,
        File::open(first_path.as_path()).unwrap()
    );

    input_wt.read(WavReader::new(&mut reader).unwrap());
    input_wt.resample_into(&mut output_wt, &mut fft_data);

    let mut writer = BufWriter::with_capacity(
        output_frame_count * output_frame_len * 4 + 1024,
        File::create(first_path.as_path()).unwrap()
    );

    output_wt.write(WavWriter::new(&mut writer, SPEC).unwrap());

    for entry in paths {

        let file_path = entry.path();

        println!("{file_path:?}");

        *reader.get_mut() = File::open(file_path.as_path()).unwrap();
    
        input_wt.read(WavReader::new(&mut reader).unwrap());
        input_wt.resample_into(&mut output_wt, &mut fft_data);
    
        *writer.get_mut() = File::create(file_path.as_path()).unwrap();
    
        output_wt.write(WavWriter::new(&mut writer, SPEC).unwrap());
    }
}