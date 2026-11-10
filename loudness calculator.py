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

from mosqito.sq_metrics import loudness_zwst
from mosqito.sq_metrics import sharpness_din_from_loudness


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
            # SAVE SUMMARY
            # ==========================================

            summary_df = pd.DataFrame([
                {
                    "File": Path(self.filename).name,
                    "Loudness_Sone": round(float(N), 2),
                    "Phon": round(float(phon), 2),
                    "Sharpness_Acum": round(float(sharpness), 2),
                    "SPL_dBA": round(float(dba), 2)
                }
            ])

            summary_file = Path(
                self.filename
            ).with_suffix(
                ".summary.csv"
            )

            summary_df.to_csv(
                summary_file,
                index=False
            )

            self.result_text.insert(
                tk.END,
                f"Summary saved:\n{summary_file}\n\n"
            )

            # ==========================================
            # LOUDNESS PROFILE
            # ==========================================

            self.compute_profile(
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

            self.result_text.insert(
                tk.END,
                "\nAnalysis Complete.\n"
            )

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

        plt.show()

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

        plt.show()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = LoudnessAnalyzer(
        root
    )

    root.mainloop()