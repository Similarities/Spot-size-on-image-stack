
# Spot-size-on-image-stack
batch for FWHM / Gaussian spatial distribution evaluation (divergence and pointing)

- Gaussian fit for beamwaist and Pointing 

How it works: 
Reads images, determines position of maximum via center of mass method  (maximum pix -> roi of 100 px compared in binned mode), creates lineout in vertical and horizontal through max position, Gaussian fit of both -> determines sigma, center, amp vertical and horizontal. 
Batch_list does this for each picture in folder and determines from center-position mean value of position and calculates for each
center-position deviation from mean value. Batch_list includes save and plots for sigma and pointing.

pointing distribution (how the pointing is distributed over space for long shot series) 
![210611_d_v](https://user-images.githubusercontent.com/40790174/123233918-c51e7a00-d4da-11eb-89aa-9f4e75dc8e8b.png)
![20210611_b_pointing](https://user-images.githubusercontent.com/40790174/123233932-c94a9780-d4da-11eb-9fbb-8bfa37fd7277.png)![20210611_d_pointing_distribution](https://user-images.githubusercontent.com/40790174/123234020-dff0ee80-d4da-11eb-87d5-39f7a7dbb8b9.png)

![20210611_c_Pointing_div_](https://user-images.githubusercontent.com/40790174/123233991-d9627700-d4da-11eb-85e0-3493ccc64879.png)
