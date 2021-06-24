# Spot-size-on-image-stack
batch for FWHM / Gaussian spatial distribution evaluation

- Gaussian fit for beamwaist and Pointing 

How it works: 
Reads images, determines position of maximum via center of mass method  (maximum pix -> roi of 100 px compared in binned mode), creates lineout in vertical and horizontal through max position, Gaussian fit of both -> determines sigma, center, amp vertical and horizontal. 
Batch_list does this for each picture in folder and determines from center-position mean value of position and calculates for each
center-position deviation from mean value. Batch_list includes save and plots for sigma and pointing.

pointing distribution (how the pointing is distributed over space for long shot series) 
