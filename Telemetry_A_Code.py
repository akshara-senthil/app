
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('telemetry_A.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

v = df["velocity_ms"]
df.loc[v < 0, "velocity_ms"] = np.nan

df["velocity_ms"] = df["velocity_ms"].interpolate(method='linear', limit_direction='both')

df = df.groupby("timestamp", as_index=False)[['velocity_ms', 'Gradient_deg', 'latitude', 'longitude']].mean()

df['t_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
df = df.interpolate(method='linear')
window = 5
df['v_smooth'] = df['velocity_ms'].rolling(window=window, center=True,min_periods=1).mean()
df['grad_rad_smooth'] = np.radians(df['Gradient_deg'].rolling(window=window, center=True,min_periods=1).mean())   

dv = np.diff(df['v_smooth'])
dt = np.diff(df['t_sec'])
accel = dv / dt


v_mid = df['v_smooth'].values[:-1]
theta = df['grad_rad_smooth'].values[:-1]

m = 300      # mass of car + driver (assumed)
g = 9.81     # Gravity
rho = 1.225  # Air density

Y = -accel - g * np.sin(theta)

A1 = g * np.cos(theta)
A2 = (rho * v_mid**2) / (2 * m)
A = np.vstack([A1, A2]).T

(crr, cda), residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)

print(f"Estimated CRR: {crr:.5f}")
print(f"Estimated CDA: {cda:.5f}")

fitted_Y = A.dot(np.array([crr, cda]))

plt.figure(figsize=(10, 6))
plt.scatter(v_mid, Y, color='blue', alpha=0.3, label='Actual Data (Decel - Gravity)')
plt.plot(v_mid, fitted_Y, color='red', linewidth=2, label='Fitted Model')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Normalized Resistive Force (m/s²)')
plt.title('Coast-down Curve Fitting: $C_{rr}$ and $C_d A$ Estimation')
plt.legend()
plt.grid(True)
plt.savefig('coast_down_curve.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved as coast_down_curve.png")