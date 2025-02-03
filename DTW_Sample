
import numpy as np
from dtw import *
from dtw import dtw
import matplotlib.pyplot as plt
import pandas as pd

#Reference: https://dynamictimewarping.github.io/python/


fp1="Kinesthetic_teaching_take_1.csv"
fp2="Kinesthetic_teaching_take_3.csv"

df1=pd.read_csv(fp1)
df2=pd.read_csv(fp2)

# read column data 'EE Position x' from csv file

column_data1= df1["EE Position x"].values
column_data2= df2["EE Position x"].values

x=np.array(column_data1)
y=np.array(column_data2)

## Display the warping curve, i.e. the alignment curve
alignment = dtw(x, y, keep_internals=True)
alignment.plot(type="threeway")

#dtw(x, y, keep_internals=True,
  #  step_pattern=rabinerJuangStepPattern(6, "d"))\
  # .plot(type="twoway",offset=-2)

plt.show()

# Save the aligned data to a CSV file

aligned_indices_1 = alignment.index1  # Indices from sequence 1
aligned_indices_2 = alignment.index2  # Indices from sequence 2

aligned_x = x[aligned_indices_1]  # Aligned sequence 1 values
aligned_y = y[aligned_indices_2]  # Aligned sequence 2 values

aligned_data = pd.DataFrame({
    "Aligned_X": aligned_x,
    "Aligned_Y": aligned_y,
    "Index_X": aligned_indices_1,
    "Index_Y": aligned_indices_2
})

# Extract aligned data as csv format

aligned_data.to_csv("aligned_trajectory_data.csv", index=False)

# Plot Query Index & Reference Index

aligned_data1 = pd.read_csv("aligned_trajectory_data.csv")

# Plot the alignment
plt.figure(figsize=(10, 5))
plt.plot(aligned_data1["Index_X"], aligned_data1["Aligned_X"], label="Query Index", linestyle="--")
plt.plot(aligned_data1["Index_Y"], aligned_data1["Aligned_Y"], label="Reference Index", linestyle="-")
plt.xlabel("Index")
plt.ylabel("EE Position x")
plt.legend()
plt.title("DTW Alignment of Trajectories (X Position)")
plt.show()
