"""Queue contact process spectra over a grid of infection rates and diffusion probabilities."""

import os

import numpy as np

LENGTH = 128
N_STEPS = 500
N_SAMPLES = 100
N_RUNS = 1000
OUTPUT_DIR = "data/contact_process/1d/all_active"

BASE_COMMAND = "cargo run --release --bin artsys --"

alpha_vals = np.linspace(1, 6, 101)
gamma_vals = [0.6, 0.7, 0.8, 0.9, 1.0]

for alpha in alpha_vals:
    for gamma in gamma_vals:
        artsys_command = " ".join(
            [
                BASE_COMMAND,
                "contact-process",
                "--dim 1",
                f"--length {LENGTH}",
                f"--alpha {alpha}",
                f"--gamma {gamma}",
                "--init all-active",
                f"--n-steps {N_STEPS}",
                f"--n-samples {N_SAMPLES}",
                f"--n-runs {N_RUNS}",
                "--emit spectrum",
                f"--dir {OUTPUT_DIR}",
                "--no-overwrite",
            ]
        )
        command = f"pueue add -- {artsys_command}"
        print(command)
        os.system(command)
