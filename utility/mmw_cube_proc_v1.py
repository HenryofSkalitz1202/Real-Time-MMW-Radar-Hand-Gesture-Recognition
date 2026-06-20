import numpy as np
from scipy.signal.windows import blackmanharris
from utility.helper import read_uint12, split_samples, parse_radar_cfg
from utility.FFTW import FFTWProcessor

class CubeProcessor:
    def __init__(self, setting, num_doppler_bin=0, num_range_bin=0, min_range=0.2, threads=4, mti_alpha=None, Doppler_to_speed=True):
        self.radar_param = parse_radar_cfg(setting)

        num_doppler_bin = max(num_doppler_bin, self.radar_param["num_chirps_per_frame"])
        num_range_bin = max(num_range_bin, self.radar_param["num_samples_per_chirp"])
        range_bin = np.arange(num_range_bin>>1) * (3e8 / (2 * self.radar_param["bandwidth"]))
        self.range_skip = np.searchsorted(range_bin, min_range)

        if Doppler_to_speed:
            doppler_bin = -np.fft.fftshift(np.fft.fftfreq(num_doppler_bin, 1 / self.radar_param["chirp_rate"]))/2*3e8/60e9
        else:
            doppler_bin = -np.fft.fftshift(np.fft.fftfreq(num_doppler_bin, 1 / self.radar_param["chirp_rate"]))

        self.proc_param = {
            "num_doppler_bin": num_doppler_bin,
            "num_range_bin": num_range_bin,
            "min_range": min_range,
            "doppler_bin": doppler_bin,
            "range_bin": range_bin
        }

        self.mti_alpha = mti_alpha

        # BGT60TR13C L-Shape Array Mapping
        # Index 0: RX3 (Corner - Reference)
        # Index 1: RX1 (Azimuth pair with RX3)
        # Index 2: RX2 (Elevation pair with RX3)
        self.active_antennas = [3, 1, 2] 

        # We ONLY perform FFT on Axis 0 (Doppler) and Axis 1 (Range)
        self.fftw_proc = FFTWProcessor(
            (self.proc_param["num_doppler_bin"], self.proc_param["num_range_bin"], 3),
            axes=(0, 1),
            precision='float32',
            threads=threads
        )

        # REVISI EVALUASI 1: Generate Blackman-Harris window 2D di saat inisialisasi agar hemat CPU
        window_doppler = blackmanharris(self.radar_param["num_chirps_per_frame"])
        window_range = blackmanharris(self.radar_param["num_samples_per_chirp"])
        self.window_2d = np.outer(window_doppler, window_range)[..., np.newaxis] # Shape: (Chirps, Samples, 1)

        self.previous_data_cube = None
        self.complex_cube = None
        self.power_rdm = None

    def mti_process(self, data_cube):
        if self.previous_data_cube is None:
            self.previous_data_cube = np.copy(data_cube)
            return np.copy(data_cube)
        else:
            mti_cube = data_cube - self.previous_data_cube
            self.previous_data_cube = self.mti_alpha * data_cube + (1 - self.mti_alpha) * self.previous_data_cube
            return mti_cube

    def process_raw_data(self, raw_data):
        adc_data = read_uint12(raw_data)
        adc_data_split = split_samples(adc_data, 1, self.radar_param["num_chirps_per_frame"], self.radar_param["num_samples_per_chirp"], self.radar_param["num_antennas"])

        # data_cube shape: (Doppler, Range, 3 Antennas)
        data_cube = np.zeros((self.proc_param["num_doppler_bin"], self.proc_param["num_range_bin"], 3), dtype=np.float32)
        
        for idx, antenna in enumerate(self.active_antennas):
            if self.radar_param["rx_mask"] & (1 << (antenna - 1)):
                data_cube[0:self.radar_param["num_chirps_per_frame"], 0:self.radar_param["num_samples_per_chirp"], idx] = adc_data_split[0, :, :, antenna - 1]

        # REVISI EVALUASI 2: Mean subtraction (penghapusan komponen DC) pada sumbu chirp untuk sinyal IF mentah
        data_cube[0:self.radar_param["num_chirps_per_frame"], 0:self.radar_param["num_samples_per_chirp"], :] -= np.mean(
            data_cube[0:self.radar_param["num_chirps_per_frame"], 0:self.radar_param["num_samples_per_chirp"], :], axis=0
        )

        if self.mti_alpha is not None:
            data_cube = self.mti_process(data_cube)

        # REVISI EVALUASI 1: Terapkan window Blackman-Harris tepat sebelum mengeksekusi 2D FFT
        data_cube[0:self.radar_param["num_chirps_per_frame"], 0:self.radar_param["num_samples_per_chirp"], :] *= self.window_2d

        self.fftw_proc.input_array[:, :, :] = data_cube
        data_cube_fft = self.fftw_proc.run()
        
        # Shift Doppler (Axis 0)
        data_cube_fft = np.fft.fftshift(data_cube_fft, axes=0)
        
        # Cut off the negative ranges and range_skip
        self.complex_cube = data_cube_fft[:, self.range_skip:self.proc_param["num_range_bin"] >> 1, :]
        
        # Integrate absolute power across all 3 antennas to create the 2D Master Range-Doppler Map
        self.power_rdm = np.sum(np.abs(self.complex_cube), axis=2) / 3.0

    def vis_2d(self):
        # We drop the Azimuth/Elevation plots to save CPU on the Raspberry Pi
        return self.power_rdm.T