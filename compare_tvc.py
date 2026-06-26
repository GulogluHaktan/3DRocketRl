import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to python path to import gimbal env
project_root = "/home/haktan/Documents/projects/RL/MuJoCo/3drocket"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mujoco
from gimbal.env import GimbalEnv
from stable_baselines3 import TD3

def load_pid_log(csv_path):
    pid_data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ang_x = float(row['angleX'])
            ang_y = float(row['angleY'])
            srv_x = float(row['servoX'])
            srv_y = float(row['servoY'])
            pid_data.append({
                'elapsed_s': float(row['elapsed_s']),
                'angleX': ang_x,
                'angleY': ang_y,
                'servoX': srv_x,
                'servoY': srv_y,
                'tilt': np.sqrt(ang_x**2 + ang_y**2),
                'servoX_dev': srv_x - 90.0,
                'servoY_dev': srv_y - 90.0,
            })
    return pid_data

def run_rl_trajectory(model_path, initial_angle_x, initial_angle_y, duration_s, control_rate=50.0):
    env = GimbalEnv(
        stage=1,  # stage 1 has no disturbances for a clean step response
        control_rate=control_rate,
        fixed_thrust_power=0.35,
        max_tvc_deg=20.0,
        tvc_servo_sec_per_60deg=0.13,
        max_tvc_rate_dps=120.0,
        max_steps=int(duration_s * control_rate) + 10,
        domain_randomization=False,
    )
    
    model = TD3.load(model_path, env=env)
    
    # Reset env and force initial conditions
    obs, _ = env.reset(seed=42)
    
    # Set the initial angle (roll = angleX, pitch = angleY)
    env.data.qpos[env.roll_qpos] = np.deg2rad(initial_angle_x)
    env.data.qpos[env.pitch_qpos] = np.deg2rad(initial_angle_y)
    env.data.qvel[env.roll_qvel] = np.deg2rad(6.2) # approximate initial velocity from log
    env.data.qvel[env.pitch_qvel] = np.deg2rad(29.3)
    
    env._sync_tvc_state()
    mujoco.mj_forward(env.model, env.data)
    
    # Re-initialize observation history with the forced initial conditions
    raw_obs = env._raw_observation()
    env.observation_history.clear()
    env.observation_history.append(raw_obs)
    env.observation_history.append(raw_obs)
    
    obs = env._observation()
    
    rl_data = []
    
    steps = int(duration_s * control_rate)
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        roll_deg = np.rad2deg(env.data.qpos[env.roll_qpos])
        pitch_deg = np.rad2deg(env.data.qpos[env.pitch_qpos])
        tilt_deg = info["tilt_deg"]
        tvc_angle_deg = info["tvc_angle_deg"]
        
        rl_data.append({
            'elapsed_s': step * env.control_dt,
            'angleX': roll_deg,
            'angleY': pitch_deg,
            'tilt': tilt_deg,
            'servoX_dev': tvc_angle_deg[0],
            'servoY_dev': tvc_angle_deg[1],
        })
            
    env.close()
    return rl_data

def trapz(y, x):
    total = 0.0
    for i in range(len(x) - 1):
        dx = x[i+1] - x[i]
        mean_y = (y[i] + y[i+1]) / 2.0
        total += dx * mean_y
    return total

def calculate_metrics(data, controller_name):
    tilts = [r['tilt'] for r in data]
    times = [r['elapsed_s'] for r in data]
    servo_x = [r['servoX_dev'] for r in data]
    servo_y = [r['servoY_dev'] for r in data]
    
    # Settling time: time after which tilt remains under 2.0 degrees
    stable_threshold = 2.0
    stable_indices = [i for i, t in enumerate(tilts) if t > stable_threshold]
    if len(stable_indices) == 0:
        settling_time = 0.0
    else:
        last_unstable_idx = stable_indices[-1]
        if last_unstable_idx < len(data) - 1:
            settling_time = times[last_unstable_idx]
        else:
            settling_time = float('inf') # never settled
            
    max_overshoot = max(tilts)
    
    # Control effort: Integral of absolute servo commands
    total_control_effort = trapz(np.abs(servo_x), times) + trapz(np.abs(servo_y), times)
    
    # Steady state error (mean tilt in the last 20% of the trajectory)
    last_20_start = int(len(data) * 0.8)
    steady_state_error = np.mean(tilts[last_20_start:])
    
    return {
        'Controller': controller_name,
        'Max Overshoot (deg)': max_overshoot,
        'Settling Time (s)': settling_time,
        'Steady State Error (deg)': steady_state_error,
        'Control Effort (deg*s)': total_control_effort
    }


def main():
    pid_csv_path = "/home/haktan/Downloads/tvc_log_20260622_144658.csv"
    model_path = "/home/haktan/Documents/projects/RL/MuJoCo/3drocket/runs/td3_gimbal_20260618_164808/td3_gimbal_best.zip"
    output_image = "/home/haktan/.gemini/antigravity-cli/brain/93b8c671-385f-4379-9d95-e6158c51a63e/tvc_comparison.png"
    
    # Load data
    pid_data = load_pid_log(pid_csv_path)
    duration_s = max(r['elapsed_s'] for r in pid_data)
    
    rl_data = run_rl_trajectory(model_path, pid_data[0]['angleX'], pid_data[0]['angleY'], duration_s)
    
    # Calculate metrics
    pid_metrics = calculate_metrics(pid_data, "PID Controller")
    rl_metrics = calculate_metrics(rl_data, "TD3 RL Controller")
    
    print(f"{'Controller':<20} | {'Overshoot':<10} | {'Settling Time':<13} | {'SSE':<10} | {'Control Effort':<15}")
    print("-" * 78)
    for m in (pid_metrics, rl_metrics):
        settle_str = f"{m['Settling Time (s)']:.2f}s" if m['Settling Time (s)'] != float('inf') else "N/A"
        print(f"{m['Controller']:<20} | {m['Max Overshoot (deg)']:.2f}° | {settle_str:<13} | {m['Steady State Error (deg)']:.2f}° | {m['Control Effort (deg*s)']:.2f}")
    
    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    fig.suptitle('Gimbal TVC Kontrolcü Karşılaştırması: PID vs TD3 RL', fontsize=16, fontweight='bold', color='#1e293b')
    
    # Color scheme
    pid_color = '#ef4444' # Red
    rl_color = '#3b82f6'  # Blue
    grid_color = '#e2e8f0'
    
    # Extract columns for plotting
    pid_time = [r['elapsed_s'] for r in pid_data]
    pid_ang_x = [r['angleX'] for r in pid_data]
    pid_ang_y = [r['angleY'] for r in pid_data]
    pid_tilt = [r['tilt'] for r in pid_data]
    pid_srv_x = [r['servoX_dev'] for r in pid_data]
    pid_srv_y = [r['servoY_dev'] for r in pid_data]
    
    rl_time = [r['elapsed_s'] for r in rl_data]
    rl_ang_x = [r['angleX'] for r in rl_data]
    rl_ang_y = [r['angleY'] for r in rl_data]
    rl_tilt = [r['tilt'] for r in rl_data]
    rl_srv_x = [r['servoX_dev'] for r in rl_data]
    rl_srv_y = [r['servoY_dev'] for r in rl_data]
    
    # 1. Angle X (Roll)
    axs[0, 0].plot(pid_time, pid_ang_x, label='Hassas Donanım PID', color=pid_color, alpha=0.8, linewidth=2)
    axs[0, 0].plot(rl_time, rl_ang_x, label='TD3 RL (Simülasyon)', color=rl_color, alpha=0.9, linewidth=2)
    axs[0, 0].set_title('Açı X (Roll) Karşılaştırması', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('Derece (°)', fontsize=10)
    axs[0, 0].grid(True, linestyle='--', color=grid_color)
    axs[0, 0].legend()
    
    # 2. Angle Y (Pitch)
    axs[0, 1].plot(pid_time, pid_ang_y, label='Hassas Donanım PID', color=pid_color, alpha=0.8, linewidth=2)
    axs[0, 1].plot(rl_time, rl_ang_y, label='TD3 RL (Simülasyon)', color=rl_color, alpha=0.9, linewidth=2)
    axs[0, 1].set_title('Açı Y (Pitch) Karşılaştırması', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('Derece (°)', fontsize=10)
    axs[0, 1].grid(True, linestyle='--', color=grid_color)
    axs[0, 1].legend()
    
    # 3. Total Tilt
    axs[1, 0].plot(pid_time, pid_tilt, label='Hassas Donanım PID', color=pid_color, alpha=0.8, linewidth=2)
    axs[1, 0].plot(rl_time, rl_tilt, label='TD3 RL (Simülasyon)', color=rl_color, alpha=0.9, linewidth=2)
    axs[1, 0].axhline(y=2.0, color='gray', linestyle=':', label='2° Stabil Sınır')
    axs[1, 0].set_title('Toplam Sapma Açısı (Tilt)', fontsize=12, fontweight='bold')
    axs[1, 0].set_ylabel('Derece (°)', fontsize=10)
    axs[1, 0].grid(True, linestyle='--', color=grid_color)
    axs[1, 0].legend()
    
    # 4. Servo X Command
    axs[1, 1].plot(pid_time, pid_srv_x, label='PID (ServoX - 90)', color=pid_color, alpha=0.8, linewidth=2)
    axs[1, 1].plot(rl_time, rl_srv_x, label='TD3 RL TVC X', color=rl_color, alpha=0.9, linewidth=2)
    axs[1, 1].set_title('Servo X Defleksiyon Komutu', fontsize=12, fontweight='bold')
    axs[1, 1].set_ylabel('Derece (°)', fontsize=10)
    axs[1, 1].grid(True, linestyle='--', color=grid_color)
    axs[1, 1].legend()
    
    # 5. Servo Y Command
    axs[2, 0].plot(pid_time, pid_srv_y, label='PID (ServoY - 90)', color=pid_color, alpha=0.8, linewidth=2)
    axs[2, 0].plot(rl_time, rl_srv_y, label='TD3 RL TVC Y', color=rl_color, alpha=0.9, linewidth=2)
    axs[2, 0].set_title('Servo Y Defleksiyon Komutu', fontsize=12, fontweight='bold')
    axs[2, 0].set_ylabel('Derece (°)', fontsize=10)
    axs[2, 0].set_xlabel('Zaman (s)', fontsize=10)
    axs[2, 0].grid(True, linestyle='--', color=grid_color)
    axs[2, 0].legend()
    
    # 6. Performance Summary Table
    axs[2, 1].axis('off')
    table_data = []
    columns = ['Metrik', 'Donanım PID', 'TD3 RL (Sim)']
    metrics_keys = [
        ('Maksimum Aşım (Overshoot)', 'Max Overshoot (deg)', '{:.2f}°'),
        ('Oturma Süresi (Settling Time)', 'Settling Time (s)', '{:.2f}s'),
        ('Kalıcı Durum Hatası (SSE)', 'Steady State Error (deg)', '{:.2f}°'),
        ('Toplam Kontrol Eforu', 'Control Effort (deg*s)', '{:.2f}')
    ]
    for label, col, fmt in metrics_keys:
        p_val = pid_metrics[col]
        r_val = rl_metrics[col]
        p_str = fmt.format(p_val) if p_val != float('inf') else "N/A"
        r_str = fmt.format(r_val) if r_val != float('inf') else "N/A"
        table_data.append([label, p_str, r_str])
        
    table = axs[2, 1].table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.5)
    
    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1e293b')
        else:
            cell.set_facecolor('#f8fafc' if row % 2 == 0 else 'white')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_image, dpi=200)
    print(f"Plot saved to: {output_image}")

if __name__ == "__main__":
    main()
