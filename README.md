# SFND-Radar-CFAR2D
## 2D CFAR Implementation
Input: 2D fft with range and dopler axis representing a radar reading.
Output: Filtered input isolating target peaks

Process:
  -Step 1: Initialize parameters for CFAR 2D (snr, guard and training window size)
  -Step 2: Loops over each cell (excluding the guard+training around the edges)
    -Step 2.1: Sum all cells values of a target cell and its surrounding including guard and training cells
    -Step 2.2: Summ all cells values of a target cell and its surrouding up to its guard cells
    -Step 2.3: Subtract Step2.2 from Step 2.1 to the sum values of only training cells
    -Step 2.4: divide the result by the number of training cells and add the SNR offset to it, resulting in the background noise
    -Step 2.5: Check if the current cell value is higher than the background noise, if yes, add it as valid target
    
## Parameters

Traning window must be bigger than guard in order to be valid, and the as it increases in value more we lose information around the edges of the image, but if it is too small it can lead for too much local sensitivity.

After trying and checking the performance of several values, the following were choosen for this application.

Range_training: 8
Range_guard: 8

Dopler_training: 4
Dopler_guard: 4

SNR gives a offset from the background noise, in order to reduce the risk of accepting values just over the background noise as valid targets.

In this case a fixed value of 10 dB was selected after checking the performance.
A dynamic value, e.g. an increment of 20% over the background noise also produced good results.

SNR : 10

## Edge cells treatment

Edges around the images, part of the training and guard cells, were not considered as valid cells thus became filtered.
