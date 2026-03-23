import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/skip/cam2_S9_Directions/moveEnet_keypoints.csv")

print(df.head())

frame_conf = df.groupby("frame")["confidence"].mean()

print(frame_conf)

df_filtered = df[df["joint"].isin([0, 1, 2])]

avg_conf_012 = df_filtered.groupby("frame")["confidence"].mean()

plt.figure(figsize=(10,5))
plt.plot(frame_conf.index, frame_conf.values, marker="o", label="All Joints")
plt.xlabel("Frame")
plt.ylabel("Average Confidence")
plt.title("Average Pose Confidence Over Frames")
plt.grid(True)
plt.tight_layout()
plt.show()
