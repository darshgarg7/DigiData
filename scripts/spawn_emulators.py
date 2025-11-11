# Copyright (c) Meta Platforms, Inc. and affiliates.

import subprocess
import signal
import os
import time
import sys
import argparse  # new import

# List to store subprocess instances
processes = []

def signal_handler(sig, frame):
    print("\nTerminating all emulators...")
    for p in processes:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception as e:
            print(f"Error terminating process {p.pid}: {e}")
    sys.exit(0)

# Register ctrl+c signal handler
signal.signal(signal.SIGINT, signal_handler)

# Parse command-line argument for number of emulators and window rendering option
parser = argparse.ArgumentParser(description="Spawn multiple emulators")
parser.add_argument("--num", type=int, default=3, help="Number of emulators to launch")
parser.add_argument("--no-window", action="store_true", help="Run emulator without rendering window")
args = parser.parse_args()
num_emulators = args.num

# Command to launch an emulator instance
command = [
    "emulator", "-avd", "DigiData",
    "-skip-adb-auth", "-no-boot-anim",
    "-gpu", "auto", "-no-skin",
    "-read-only", "-no-cache", "-no-snapshot",
    "-no-audio", "-cores", "2",
    "-memory", "2048",
    "-lowram", "-no-passive-gps",
    "-no-location-ui", "-no-metrics",
    "-no-nested-warnings"
]
if args.no_window:  # Append no-window flag if specified
    command.append("-no-window")

for i in range(num_emulators):  # use num_emulators instead of hardcoded 10
    print(f"Launching emulator instance {i+1}")
    p = subprocess.Popen(command, preexec_fn=os.setsid)
    processes.append(p)
    time.sleep(10)

# Wait indefinitely until processes finish or ctrl+c is pressed
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    signal_handler(None, None)