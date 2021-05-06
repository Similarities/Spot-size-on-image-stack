# Spot-size-on-image-stack
batch for FWHM / Gaussian spatial distribution evaluation

- Gaussian fit for beamwaist and Pointing 

How it works: 
Reads images, determines position of max  (simple single picture), creates lineout in vertical and horizontal through max position, Gaussian fit of both -> determines sigma vertical and horizontal
, determines center of Gaussian fit in both directions. Batch_list does this for each picture in folder and determines from center-position mean value of position and calculates for each
center-position deviation from mean value. Batch_list includes save and plots for sigma and pointing.
