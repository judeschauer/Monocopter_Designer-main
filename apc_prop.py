import requests
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import io
import os
import re

def load_apc_prop_data(prop_code, cache_dir="apc_data/PERFILES_WEB/PERFILES2"):
    """
    Load APC prop data:
    - If file exists locally, read it
    - Otherwise download and cache it

    Returns:
        dict: rpm_value -> DataFrame
    """
    os.makedirs(cache_dir, exist_ok=True)

    filename = f"{prop_code}.dat"
    filepath = os.path.join(cache_dir, filename)

    # Step 1: Load from cache if exists
    if os.path.exists(filepath):
        print(f"Found: {filename}")
        with open(filepath, "r") as f:
            text = f.read()
    else:
        print(f"Couldn't find file: {filename}")

    # Step 3: Parse RPM blocks
    rpm_blocks = {}
    current_rpm = None
    lines = []

    for line in text.splitlines():
        m = re.match(r"\s*PROP RPM\s*=\s*(\d+)", line)

        if m:
            if current_rpm is not None and lines:
                rpm_blocks[current_rpm] = _parse_block(lines)

            current_rpm = int(m.group(1))
            lines = []
        else:
            if current_rpm is not None:
                lines.append(line)

    # Final block
    if current_rpm is not None and lines:
        rpm_blocks[current_rpm] = _parse_block(lines)

    return rpm_blocks

def _parse_block(lines):
    # Skip empty lines
    lines = [l for l in lines if l.strip()]

    # First two lines are headers
    header1 = lines[0].split()
    header2 = lines[1].split()

    # Combine them
    header = [h1 + " " + h2 for h1, h2 in zip(header1, header2)]

    # Data lines start after header lines
    data_lines = lines[2:]

    # Split rows
    data = [row.split() for row in data_lines if row.strip()]

    df = pd.DataFrame(data, columns=header).apply(pd.to_numeric, errors="ignore")
    return df

def get_prop_data_at_rpm(prop_rpm, prop_code):
    """
    Return the DataFrame for the specified PROP_RPM.
    """
    data = load_apc_prop_data(prop_code)
    if prop_rpm not in data:
        raise ValueError(f"RPM {prop_rpm} not found in data file.")
    return data[prop_rpm]

def get_motor_thrust(prop_code, motor_rpm, axial_speed_requested):
    """
    Returns linearly interpolated thrust (N) at the requested axial speed (m/s)
    for the given propeller and RPM.
    
    apc_prop_selection : URL or local filename for APC prop data
    motor_rpm : RPM of the propeller
    axial_speed_requested : axial velocity in m/s
    """
    # Load the RPM block
    df = get_prop_data_at_rpm(motor_rpm, prop_code)
    
    # Extract axial speeds (m/s) and thrusts (N)
    axial_speeds = df["V (mph)"] * 0.44704  # convert mph to m/s
    thrusts = df["Thrust (N)"]
    
    # Linear interpolation
    thrust_interp = np.interp(axial_speed_requested, axial_speeds, thrusts)
    
    return thrust_interp

def get_thrusts_at_rpm(prop_code, axial_speeds, motor_rpm):
    # Load the RPM block
    df = get_prop_data_at_rpm(motor_rpm, prop_code)
    
    # Extract axial speeds (m/s) and thrusts (N)
    axial_speeds = df["V (mph)"] * 0.44704  # convert mph to m/s
    thrusts = df["Thrust (N)"]
    
    # Linear interpolation
    thrusts_interp = []
    for axial_speed_requested in axial_speeds:
        thrust_interp = np.interp(axial_speed_requested, axial_speeds, thrusts)
        thrusts_interp.append(thrust_interp)

    return thrusts_interp

def get_motor_dict(prop_code, motor_rpm_range, requested_axial_speeds_dict):
    data = load_apc_prop_data(prop_code)

    combined_motor_thrusts_dict = {}
    for r_motor, axial_speeds_requested in requested_axial_speeds_dict.items():    

        motor_thrusts_dict = {}

        for motor_rpm in motor_rpm_range:

            if motor_rpm not in data:
                raise ValueError(f"RPM {motor_rpm} not found in data file.")
            perf_table_at_rpm = data[motor_rpm]


            # Extract axial speeds (m/s) and thrusts (N)
            axial_speeds = perf_table_at_rpm["V (mph)"] * 0.44704  # convert mph to m/s
            thrusts = perf_table_at_rpm["Thrust (N)"]

            # Linear interpolation
            thrust_interp = np.interp(axial_speeds_requested, axial_speeds, thrusts)
            motor_thrusts_dict[motor_rpm] = thrust_interp

        combined_motor_thrusts_dict[r_motor] = motor_thrusts_dict

    return combined_motor_thrusts_dict

def get_prop_perf_dict(prop_code, motor_rpm_range, axial_speed_range):
    data = load_apc_prop_data(prop_code)
    prop_performance_dict = {}
    for motor_rpm in motor_rpm_range:

        if motor_rpm not in data:
            raise ValueError(f"RPM {motor_rpm} not found in data file.")
        perf_table_at_rpm = data[motor_rpm]


        # Extract axial speeds (m/s) and thrusts (N)
        axial_speeds = perf_table_at_rpm["V (mph)"] * 0.44704  # convert mph to m/s
        thrusts = perf_table_at_rpm["Thrust (N)"]
        powers = perf_table_at_rpm["PWR (W)"]
        efficiency = perf_table_at_rpm["Pe -"]

        # Linear interpolation
        thrust_interp = np.interp(axial_speed_range, axial_speeds, thrusts)
        power_interp = np.interp(axial_speed_range, axial_speeds, powers)
        efficiency_interp = np.interp(axial_speed_range, axial_speeds, efficiency)
        prop_performance_dict[motor_rpm] = [thrust_interp, power_interp, efficiency_interp]

    return prop_performance_dict

def get_prop_km(prop_code):
    data = load_apc_prop_data(prop_code)
    