import os
import numpy as np


LENGTH = 128
ALPHA_CRIT = 3.29785
N_STEPS = 512
N_SAMPLES = 100_000

BASE_COMMAND = "cargo run --bin contact_process_1d --release --"

rate_vals = np.linspace(1, 6, 101)
diffusion_vals = [0.6, 0.7, 0.8, 0.9, 1.0]
# diffusion_vals = np.linspace(0, 1, 11)

for rate in rate_vals:
    for diffusion in diffusion_vals:
        output = f"data/contact_process/new/contact-process-1d-diffusion_L={LENGTH}_rate={rate:.8}_diffusion={diffusion:.8}_n_steps={N_STEPS}_n_samples={N_SAMPLES}"
        cargo_command = " ".join(
            [
                BASE_COMMAND,
                f"--length {LENGTH}",
                f"--n-steps {N_STEPS}",
                f"--n-samples {N_SAMPLES}",
                f"--rate {rate}",
                f"--diffusion {diffusion}",
                f"--output {output}",
            ]
        )
        command = f"pueue add {cargo_command}"
        os.system(f""" echo "{command}" """)
        os.system(command)
