from PIL import Image
import numpy as np
import os

ids = os.listdir('./PATCHES/train/im1/')

class_sums = [0]*4

for id in ids:
    cat = int(id[-5])
    class_sums[cat] = class_sums[cat] + 1

for i in range(0, len(class_sums)):
    print('class {} '.format(i), class_sums[i]/len(ids))







# Count occurrences of each class
class_counts = np.array(class_sums)
total_samples = len(ids)

# Calculate class frequencies
class_freqs = class_counts / total_samples
print("Class frequencies:", class_freqs)

# Calculate alpha for each class as the inverse of class frequency
alpha = 1.0 / class_freqs
print("Alpha values:", alpha)


# Normalize alpha to sum to 1
alpha /= alpha.sum()
print("Normalized alpha values:", alpha)
