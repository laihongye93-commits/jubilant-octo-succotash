import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from tkinter import simpledialog
import json

from scipy.signal import bilinear, lfilter
from scipy.signal import welch, find_peaks
from scipy.signal import get_window

from mosqito.sq_metrics import loudness_zwst
from mosqito.sq_metrics import sharpness_din_from_loudness

from openpyxl import Workbook
from openpyxl.drawing.image import Image

from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
# =====================================================
# DIN45681 TONALITY
# =====================================================

def masking_index_av(freq):
    """
    Approximate DIN45681 masking index.
    """

    f_pts = np.array([
        90, 125, 160, 200, 250, 315,
        400, 500, 630, 800,
        1000, 1250, 1600,
        2000, 2500, 3150, 4000
    ])

    av_pts = np.array([
        8.0, 7.5, 7.0, 6.0, 5.0,
        4.0, 3.5, 3.0, 2.5,
        2.0, 1.5, 1.0, 0.5,
        0.0, 0.0, 0.0, 0.0
    ])

    return np.interp(
        freq,
        f_pts,
        av_pts,
        left=8.0,
        right=0.0
    )


def bark_bandwidth(freq):
    """
    Approximate critical-band width (Hz)
    """

    return (
        25 +
        75 *
        (1 + 1.4 * (freq / 1000.0) ** 2) ** 0.69
    )


def din45681_kt(delta_l):
    """
    Official DIN45681 Kt table
    """

    if delta_l <= 0:
        return 0
    elif delta_l <= 2:
        return 1
    elif delta_l <= 4:
        return 2
    elif delta_l <= 6:
        return 3
    elif delta_l <= 9:
        return 4
    elif delta_l <= 12:
        return 5
    else:
        return 6


def din45681_analysis(
        signal,
        fs,
        fmin=90):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    signal = signal - np.mean(signal)

    frequencies, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(
            16384,
            len(signal)
        ),
        scaling="spectrum"
    )

    psd_db = 10 * np.log10(
        psd + 1e-20
    )

    peaks, _ = find_peaks(
        psd_db,
        prominence=3.0
    )

    results = []

    for peak_idx in peaks:

        ft = frequencies[peak_idx]

        if ft < fmin:
            continue

        bw = bark_bandwidth(ft)

        band_idx = np.where(
            (frequencies >= ft - bw / 2) &
            (frequencies <= ft + bw / 2)
        )[0]

        if len(band_idx) < 5:
            continue

        Lt = psd_db[peak_idx]

        exclude = np.arange(
            max(0, peak_idx - 1),
            min(len(frequencies),
                peak_idx + 2)
        )

        noise_idx = np.setdiff1d(
            band_idx,
            exclude
        )

        if len(noise_idx) == 0:
            continue

        noise_energy = np.sum(
            10 ** (
                psd_db[noise_idx] / 10
            )
        )

        Lg = 10 * np.log10(
            noise_energy + 1e-30
        )

        av = masking_index_av(ft)

        delta_l = Lt - Lg - av

        kt = din45681_kt(delta_l)

        results.append({
            "frequency_hz": float(ft),
            "delta_l": float(delta_l),
            "kt": float(kt),
            "Lt": float(Lt),
            "Lg": float(Lg),
            "av": float(av)
        })

    return sorted(
        results,
        key=lambda x: x["delta_l"],
        reverse=True
    )

# =====================================================
# A-WEIGHTING FILTER
# =====================================================

def a_weighting(fs):

    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    nums = [
        (2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)),
        0,
        0,
        0,
        0
    ]

    dens = np.polymul(
        [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2]
    )

    dens = np.polymul(
        np.polymul(dens, [1, 2 * np.pi * f3]),
        [1, 2 * np.pi * f2]
    )

    return bilinear(nums, dens, fs)


def calculate_dba(signal, fs):

    b, a = a_weighting(fs)

    weighted = lfilter(b, a, signal)

    rms = np.sqrt(np.mean(weighted ** 2))

    if rms <= 0:
        return 0

    return 20 * np.log10(rms / 20e-6)


def sone_to_phon(sone):

    if sone <= 0:
        return 0

    return 40 + 10 * np.log2(sone)
# =====================================================
# DIN45681 TONALITY
# =====================================================

def tonality_vs_time(
        signal,
        fs,
        window_sec=3.0,
        overlap=0.5):

    frame_size = int(
        window_sec * fs
    )

    hop_size = int(
        frame_size *
        (1 - overlap)
    )

    times = []
    kt_values = []
    tone_freqs = []
    delta_values = []

    for start in range(
            0,
            len(signal) - frame_size,
            hop_size):

        frame = signal[
                start:
                start + frame_size
                ]

        tones = din45681_analysis(
            frame,
            fs
        )

        center_time = (
                start +
                frame_size / 2
        ) / fs

        times.append(center_time)

        if len(tones):

            strongest = tones[0]

            kt_values.append(
                strongest["kt"]
            )

            tone_freqs.append(
                strongest["frequency_hz"]
            )

            delta_values.append(
                strongest["delta_l"]
            )

        else:

            kt_values.append(0)

            tone_freqs.append(
                np.nan
            )

            delta_values.append(0)

    return (
        np.array(times),
        np.array(kt_values),
        np.array(tone_freqs),
        np.array(delta_values)
    )

# =====================================================
# GUI
# =====================================================

class LoudnessAnalyzer:

    def __init__(self, root):

        self.root = root

        self.root.title("PC Fan Acoustic Analyzer")

        self.root.geometry("900x900")

        self.filename = None

        tk.Label(
            root,
            text="ISO 532 Loudness Analyzer",
            font=("Arial", 16, "bold")
        ).pack(pady=10)

        self.file_label = tk.Label(
            root,
            text="No file selected",
            fg="blue"
        )

        self.file_label.pack()

        tk.Button(
            root,
            text="Select Audio File",
            width=25,
            command=self.select_file
        ).pack(pady=5)

        tk.Button(
            root,
            text="Run Analysis",
            width=25,
            command=self.start_analysis
        ).pack(pady=5)

        self.result_text = tk.Text(
            root,
            height=25
        )

        self.result_text.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )
        self.calibration_factor = 1.0
        self.peak_frequency = 0.0

        #tk.Button(
        #    root,
        #    text="Calibrate",
         #   width=25,
         #   command=self.calibrate
        #).pack(pady=5)

        #self.cal_label = tk.Label(
        #    root,
        #    text=f"Calibration Factor: {self.calibration_factor:.4f}"
        #)

        #self.cal_label.pack()
        tk.Button(
            root,
            text="Calibrate From SPL Meter",
            width=25,
            command=self.calibrate_from_spl
        ).pack(pady=5)


        try:
            with open("calibration.json", "r") as f:
                self.calibration_factor = json.load(f)["factor"]
        except:
            pass

        self.cal_label = tk.Label(
            root,
            text=f"Calibration Factor: {self.calibration_factor:.4f}"
        )

        self.cal_label.pack()

    # =================================================
    def calibrate_from_spl(self):

        cal_file = filedialog.askopenfilename(
            title="Select Recording Used For SPL Measurement",
            filetypes=[
                ("Audio Files", "*.wav *.flac *.aiff *.aif")
            ]
        )

        if not cal_file:
            return

        spl = simpledialog.askfloat(
            "Reference SPL",
            "Enter measured SPL (dB):\n\nExample: 35.2"
        )

        if spl is None:
            return

        try:

            signal, fs = sf.read(cal_file)

            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)

            rms = np.sqrt(
                np.mean(signal ** 2)
            )

            if rms <= 0:
                raise Exception(
                    "Invalid recording"
                )

            # Convert dB SPL -> Pa
            pressure_pa = (
                    20e-6 *
                    10 ** (spl / 20)
            )

            self.calibration_factor = (
                    pressure_pa / rms
            )

            with open(
                    "calibration.json",
                    "w"
            ) as f:

                json.dump(
                    {
                        "factor":
                            self.calibration_factor
                    },
                    f,
                    indent=4
                )

            self.cal_label.config(
                text=
                f"Calibration Factor: "
                f"{self.calibration_factor:.6f}"
            )

            result = (
                f"Measured SPL : {spl:.2f} dB\n"
                f"Recording RMS: {rms:.6f}\n"
                f"Pressure     : "
                f"{pressure_pa:.8f} Pa\n\n"
                f"Calibration Factor:\n"
                f"{self.calibration_factor:.6f}"
            )

            messagebox.showinfo(
                "Calibration Complete",
                result
            )

        except Exception as e:

            messagebox.showerror(
                "Calibration Error",
                str(e)
            )
    def calibrate(self):

        cal_file = filedialog.askopenfilename(
            title="Select Calibration Recording",
            filetypes=[
                ("Audio Files", "*.wav *.flac")
            ]
        )

        if not cal_file:
            return

        try:

            signal, fs = sf.read(cal_file)

            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)

            rms = np.sqrt(
                np.mean(signal ** 2)
            )

            if rms <= 0:
                raise Exception(
                    "Invalid calibration recording."
                )

            #
            # Typical acoustic calibrator:
            # 94 dB SPL @ 1 kHz
            #
            reference_pressure = 1.002

            self.calibration_factor = (
                    reference_pressure / rms
            )

            self.cal_label.config(
                text=f"Calibration Factor: "
                     f"{self.calibration_factor:.3f}"
            )

            messagebox.showinfo(
                "Calibration Complete",
                f"Factor = "
                f"{self.calibration_factor:.3f}"
            )

        except Exception as e:

            messagebox.showerror(
                "Calibration Error",
                str(e)
            )
    def select_file(self):

        self.filename = filedialog.askopenfilename(
            filetypes=[
                (
                    "Audio files",
                    "*.wav *.flac *.aiff *.aif"
                )
            ]
        )

        if self.filename:
            self.file_label.config(
                text=self.filename
            )

    # =================================================

    def start_analysis(self):

        if not self.filename:
            messagebox.showerror(
                "Error",
                "Please select a file."
            )
            return

        threading.Thread(
            target=self.analyze_file,
            daemon=True
        ).start()

    # =================================================

    def analyze_file(self):

        try:

            self.result_text.delete(
                "1.0",
                tk.END
            )

            self.result_text.insert(
                tk.END,
                "Loading file...\n\n"
            )

            signal, fs = sf.read(
                self.filename
            )
            signal = signal * self.calibration_factor

            if signal.ndim > 1:
                signal = np.mean(
                    signal,
                    axis=1
                )

            duration = len(signal) / fs

            self.result_text.insert(
                tk.END,
                f"File: {Path(self.filename).name}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Sample Rate: {fs} Hz\n"
            )

            self.result_text.insert(
                tk.END,
                f"Duration: {duration:.2f} sec\n\n"
            )

            # ==========================================
            # LOUDNESS
            # ==========================================

            N, N_specific, bark_axis = loudness_zwst(
                signal,
                fs
            )
            self.bark_axis = bark_axis
            self.N_specific = N_specific

            phon = sone_to_phon(
                float(N)
            )

            sharpness = sharpness_din_from_loudness(
                N,
                N_specific
            )

            dba = calculate_dba(
                signal,
                fs
            )

            self.result_text.insert(
                tk.END,
                "===== OVERALL RESULTS =====\n\n"
            )

            self.result_text.insert(
                tk.END,
                f"Loudness      : {N:.2f} sones\n"
            )

            self.result_text.insert(
                tk.END,
                f"Phon Level    : {phon:.2f} phon\n"
            )

            self.result_text.insert(
                tk.END,
                f"Sharpness     : {sharpness:.2f} acum\n"
            )

            self.result_text.insert(
                tk.END,
                f"SPL(A)        : {dba:.2f} dB(A)\n\n"
            )


            # ==========================================
            # LOUDNESS PROFILE
            # ==========================================

            self.compute_profile(
                signal,
                fs
            )
            tonality_png = self.plot_tonality(
                signal,
                fs
            )
            # ==========================================
            # SPECIFIC LOUDNESS PLOT
            # ==========================================

            self.plot_specific_loudness(
                bark_axis,
                N_specific
            )

            # ==========================================
            # SPECTRUM PLOT
            # ==========================================
            spectrum_png = self.plot_frequency_spectrum(
                signal,
                fs
            )

            self.result_text.insert(
                tk.END,
                "\nAnalysis Complete.\n"
            )

            # ==========================================
            # SAVE SUMMARY
            # ==========================================
            summary_file = Path(self.filename).with_suffix(".summary.xlsx")

            wb = Workbook()
            ws = wb.active
            ws.title = "Summary"

            # Summary Data
            ws["A1"] = "File"
            ws["B1"] = "Loudness (Sone)"
            ws["C1"] = "Phon"
            ws["D1"] = "Sharpness (Acum)"
            ws["E1"] = "SPL dB(A)"
            ws["F1"] = "Peak Frequency (Hz)"

            ws["A2"] = Path(self.filename).name
            ws["B2"] = round(float(N), 2)
            ws["C2"] = round(float(phon), 2)
            ws["D2"] = round(float(sharpness), 2)
            ws["E2"] = round(float(dba), 2)
            ws["F2"] = round(float(self.peak_frequency), 1)

            # Insert plots
            profile_png = Path(self.filename).with_suffix(".loudness.png")
            specific_png = Path(self.filename).with_suffix(".specific_loudness.png")
            spectrum_png = Path(self.filename).with_suffix(".frequency_spectrum.png")
            tonality_png = Path(self.filename).with_suffix(
                ".tonality.png"
            )

            if profile_png.exists():
                img1 = Image(str(profile_png))
                img1.width = 800
                img1.height = 400
                ws.add_image(img1, "A5")

            if specific_png.exists():
                img2 = Image(str(specific_png))
                img2.width = 800
                img2.height = 400
                ws.add_image(img2, "A30")

            if spectrum_png.exists():
                img3 = Image(str(spectrum_png))
                img3.width = 800
                img3.height = 400
                ws.add_image(img3, "A55")

            if tonality_png.exists():
                img4 = Image(
                    str(tonality_png)
                )

                img4.width = 800
                img4.height = 400

                ws.add_image(
                    img4,
                    "A80"
                )
            tonality_ws = wb.create_sheet(
                "Tonality"
            )

            tonality_ws.append(
                [
                    "Time (s)",
                    "Kt (dB)",
                    "Delta L (dB)",
                    "Tone Frequency (Hz)"
                ]
            )

            for t, k, d, f in zip(
                    self.tonality_axis,
                    self.tonality_time,
                    self.delta_l_time,
                    self.tone_frequency_time):
                tonality_ws.append([
                    float(t),
                    float(k),
                    float(d),
                    float(f) if not np.isnan(f) else None
                ])

            wb.save(summary_file)

            self.result_text.insert(
                tk.END,
                f"Summary saved:\n{summary_file}\n\n"
            )
            html_file = Path(self.filename).with_suffix(".summary.html")

            self.save_html_report(
                html_file,
                self.filename,
                float(N),
                float(phon),
                float(sharpness),
                float(dba),
                float(self.peak_frequency),
                self.loudness_time,
                self.time_axis,
                self.bark_axis,
                self.N_specific,
                self.frequencies,
                self.psd_db
            )

            os.startfile(html_file)

        except Exception as e:

            messagebox.showerror(
                "Analysis Error",
                str(e)
            )


    # =================================================

    def compute_profile(
            self,
            signal,
            fs):

        window = int(
            0.5 * fs
        )

        hop = int(
            0.1 * fs
        )

        times = []
        sones = []

        self.result_text.insert(
            tk.END,
            "Generating loudness profile...\n"
        )

        for start in range(
                0,
                len(signal) - window,
                hop):

            try:

                frame = signal[
                    start:start + window
                ]

                result = loudness_zwst(
                    frame,
                    fs
                )

                loudness = float(
                    result[0]
                )

                times.append(
                    start / fs
                )

                sones.append(
                    loudness
                )

            except:
                pass

        if len(sones):

            avg_sone = np.mean(
                sones
            )

            min_sone = np.min(
                sones
            )

            max_sone = np.max(
                sones
            )

            self.result_text.insert(
                tk.END,
                f"Average Loudness : {avg_sone:.2f} sones\n"
            )

            self.result_text.insert(
                tk.END,
                f"Peak Loudness    : {max_sone:.2f} sones\n"
            )

            self.result_text.insert(
                tk.END,
                f"Minimum Loudness : {min_sone:.2f} sones\n\n"
            )

        # ==========================================
        # SAVE PROFILE CSV
        # ==========================================

        profile_df = pd.DataFrame({
            "Time_s": times,
            "Loudness_Sone": sones
        })

        csv_file = Path(
            self.filename
        ).with_suffix(
            ".loudness.csv"
        )

        profile_df.to_csv(
            csv_file,
            index=False
        )

        self.result_text.insert(
            tk.END,
            f"Profile saved:\n{csv_file}\n\n"
        )

        #===========================================
        self.time_axis = times
        self.loudness_time = sones

        # ==========================================
        # PLOT
        # ==========================================

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            times,
            sones,
            color="blue"
        )

        plt.title(
            "Loudness vs Time"
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            "Loudness (sone)"
        )

        plt.grid(True)

        plt.tight_layout()

        profile_png = Path(self.filename).with_suffix(".loudness.png")
        plt.savefig(profile_png, dpi=300, bbox_inches="tight")
        plt.close()

        self.result_text.insert(
            tk.END,
            f"Loudness plot saved:\n{profile_png}\n\n"
        )

    # =================================================

    def plot_specific_loudness(
            self,
            bark_axis,
            specific_loudness):

        plt.figure(
            figsize=(10, 5)
        )

        plt.plot(
            bark_axis,
            specific_loudness,
            color="red",
            linewidth=2
        )

        plt.title(
            "Specific Loudness Distribution"
        )

        plt.xlabel(
            "Critical Band Rate (Bark)"
        )

        plt.ylabel(
            "Specific Loudness (sone/Bark)"
        )

        plt.grid(True)

        plt.tight_layout()

        specific_png = Path(self.filename).with_suffix(".specific_loudness.png")
        plt.savefig(specific_png, dpi=300, bbox_inches="tight")
        plt.close()

        self.result_text.insert(
            tk.END,
            f"Specific loudness plot saved:\n{specific_png}\n\n"
        )
    # =================================================
    def plot_frequency_spectrum(self, signal, fs):

        self.result_text.insert(
            tk.END,
            "Generating frequency spectrum...\n"
        )

        # Remove DC offset
        signal -= np.mean(signal)

        # Welch PSD
        frequencies, psd = welch(
            signal,
            fs=fs,
            window="hann",
            nperseg=min(8192, len(signal)),
            noverlap=None,
            scaling="spectrum"
        )

        # Ignore subsonic region
        valid = frequencies >= 20

        frequencies = frequencies[valid]
        psd = psd[valid]

        psd_db = 10 * np.log10(psd + 1e-20)

        # Find dominant frequency
        peak_idx = np.argmax(psd)
        peak_frequency = frequencies[peak_idx]
        self.peak_frequency = float(peak_frequency)
        self.frequencies = frequencies
        self.psd_db = psd_db


        # Tonal peaks (optional)
        peaks, _ = find_peaks(
            psd_db,
            prominence=3.0
        )

        plt.figure(figsize=(10, 5))

        plt.semilogx(
            frequencies,
            psd_db,
            color="green",
            linewidth=1.5
        )

        plt.scatter(
            frequencies[peaks],
            psd_db[peaks],
            color="red",
            s=12
        )

        plt.title("Frequency Spectrum (Welch PSD)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Level (dB)")
        plt.grid(True, which="both")

        plt.tight_layout()

        spectrum_png = Path(
            self.filename
        ).with_suffix(
            ".frequency_spectrum.png"
        )

        plt.savefig(
            spectrum_png,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        self.result_text.insert(
            tk.END,
            f"Peak Frequency : {peak_frequency:.1f} Hz\n"
        )

        self.result_text.insert(
            tk.END,
            f"Spectrum saved:\n{spectrum_png}\n\n"
        )

        return spectrum_png

    def plot_tonality(self, signal, fs):

        self.result_text.insert(
            tk.END,
            "Generating tonality profile...\n"
        )

        times, kt, tone_freq, delta_l = (
            tonality_vs_time(
                signal,
                fs
            )
        )

        self.tonality_time = kt
        self.tonality_axis = times
        self.tone_frequency_time = tone_freq
        self.delta_l_time = delta_l

        plt.figure(figsize=(10, 5))

        plt.plot(
            times,
            kt,
            color="red",
            linewidth=2
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Kt (dB)")
        plt.title(
            "DIN45681 Tonality vs Time"
        )

        plt.grid(True)

        plt.tight_layout()

        tonality_png = Path(
            self.filename
        ).with_suffix(
            ".tonality.png"
        )

        plt.savefig(
            tonality_png,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        self.result_text.insert(
            tk.END,
            f"Tonality plot saved:\n"
            f"{tonality_png}\n\n"
        )

        return tonality_png

    def save_html_report(
            self,
            output_html,
            filename,
            loudness,
            phon,
            sharpness,
            dba,
            peak_frequency,
            loudness_time,
            time_axis,
            bark_axis,
            specific_loudness,
            frequencies,
            psd_db):

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Loudness vs Time",
                "DIN45681 Tonality vs Time",
                "Frequency Spectrum (Welch PSD)",
                "Specific Loudness"
            ),
            vertical_spacing=0.10
        )

        # Loudness Profile
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=loudness_time,
                mode="lines",
                name="Loudness"
            ),
            row=1,
            col=1
        )

        fig.update_xaxes(
            title_text="Time (s)",
            row=1,
            col=1
        )

        fig.update_yaxes(
            title_text="Loudness (Sone)",
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.tonality_axis,
                y=self.tonality_time,
                mode="lines",
                name="Tonality"
            ),
            row=2,
            col=1
        )

        fig.update_xaxes(
            title_text="Time (s)",
            row=2,
            col=1
        )

        fig.update_yaxes(
            title_text="Kt (dB)",
            row=2,
            col=1
        )
        # Frequency Spectrum
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=psd_db,
                mode="lines",
                name="Spectrum"
            ),
            row=3,
            col=1
        )

        # Specific Loudness
        fig.add_trace(
            go.Scatter(
                x=bark_axis,
                y=specific_loudness,
                mode="lines",
                name="Specific Loudness"
            ),
            row=4,
            col=1
        )

        fig.update_xaxes(
            type="log",
            title_text="Frequency (Hz)",
            row=3,
            col=1
        )

        fig.update_yaxes(
            title_text="Level (dB)",
            row=3,
            col=1
        )

        fig.update_xaxes(
            title_text="Bark",
            row=4,
            col=1
        )

        fig.update_yaxes(
            title_text="Sone/Bark",
            row=4,
            col=1
        )

        fig.update_layout(
            height=1800,
            title=(
                f"{Path(filename).name}<br>"
                f"Loudness={loudness:.2f} sone | "
                f"Phon={phon:.1f} | "
                f"Sharpness={sharpness:.2f} acum | "
                f"SPL(A)={dba:.2f} dB(A) | "
                f"Peak={peak_frequency:.1f} Hz"
            )
        )

        tonality_table = """
        <h3>DIN45681 Tonality Results</h3>
        <table border="1" cellpadding="6">
        <tr>
            <th>Time (s)</th>
            <th>Kt (dB)</th>
            <th>ΔL (dB)</th>
            <th>Tone Frequency (Hz)</th>
        </tr>
        """

        for t, k, d, f in zip(
                self.tonality_axis,
                self.tonality_time,
                self.delta_l_time,
                self.tone_frequency_time):
            tonality_table += f"""
            <tr>
                <td>{t:.2f}</td>
                <td>{k:.1f}</td>
                <td>{d:.2f}</td>
                <td>{'' if np.isnan(f) else f'{f:.1f}'}</td>
            </tr>
            """

        tonality_table += "</table><br>"

        html = f"""
        <html>
        <head>
            <title>Acoustic Analysis Report</title>
        </head>
        <body>

        <h2>Acoustic Analysis Report</h2>

        <table border="1" cellpadding="6">
            <tr><td>File</td><td>{Path(filename).name}</td></tr>
            <tr><td>Loudness</td><td>{loudness:.2f} sone</td></tr>
            <tr><td>Phon</td><td>{phon:.1f}</td></tr>
            <tr><td>Sharpness</td><td>{sharpness:.2f} acum</td></tr>
            <tr><td>SPL(A)</td><td>{dba:.2f} dB(A)</td></tr>
            <tr><td>Peak Frequency</td><td>{peak_frequency:.1f} Hz</td></tr>
        </table>
        
        <br>
        
        {tonality_table}
        
        {fig.to_html(
            full_html=False,
            include_plotlyjs='cdn'
        )}
        </body>
        </html>
        """

        with open(
                output_html,
                "w",
                encoding="utf-8"
        ) as f:
            f.write(html)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = LoudnessAnalyzer(
        root
    )

    root.mainloop()