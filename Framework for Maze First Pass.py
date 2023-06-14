# Goals for this program:
# 1) Open a saved image file of a maze
# 2) Convert it to an array of pixel values
# 3) Display the image back to the user

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt













# ============ Below: code copied from what I did for Bajcsy: ==========================

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit

# def gaussian(x, coeff, mean, sd, const): # Just a generic gaussian
#     exp = np.exp(-(x - mean)**2 / (2 * sd**2))
#     f = coeff*exp + const
#     return f

# def gaussian_steck(x, coeff, mean, beam_rad, const): # Definition from Ch. 6 of Steck's Optics
#     exp = np.exp(-2 * (x - mean)**2 / (beam_rad**2)) # NOTE: now defined in terms of the intensity
#     f = coeff*exp + const
#     return f


# # ===================== Load in the image: ============================

# image1 = mpimg.imread('March 29 Image x1.tif')
# # From what I understand, the result of this will be an array of 3 dimensions: the length and
# # width of the image in pixels, and a third dimension that has 4 values per pixel.
# # From what I understand, for a .tif image, these four values will be of the RBGA format.
# # The first three values for each pixel are the R, G, and B values, and the fourth is the opacity

# print("Inputted Image:")
# image1plot = plt.imshow(image1) # Shows the image again
# plt.show()

# image1_grayscale = image1[:, :, 0] # Every R/G/B value is the same, so let's just turn this into a 2D array instead of a 3D
# # array with all of the entries being the same in one dimension. Now we have a 2D array representing a greyscale image



# # ====================== Deciding where to take the data from in the image: ========================

# # We only need a 1D slice of the image to fit to a 1D Gaussian, but it's important to make sure that this 1D slice goes
# # through the center of the beam on the camera; we need to measure the width at its widest point.
# # Previously, I did this manually, but a better way is needed.
# # I'll decide now to (arbitrarily) take a vertical slice of the image (i.e. a column of pixels). 
# # Images from this camera are 1024 px tall by 1280 px wide, so this will be a slice of 1024 px

# max_pix_val = max(image1_grayscale.flatten()) # the maximum value of a pixel in the image
# min_pix_val = min(image1_grayscale.flatten()) # the minimum value of a pixel in the image

# # Plan: find the column that has the most pixels within a certain threshold of the maximum value, and call that the middle.
# # This makes intuitive sense since we expect a slice from the middle of a Gaussian-like beam to have the greatest intensity.
# thresh_val = round(max_pix_val - max_pix_val * 0.15)
# num_columns = len(image1_grayscale[0, :])
# num_rows = len(image1_grayscale[:, 0])

# bright_pixels = [] # the ith entry stores the number of bright pixels in column i
# for column in range(0, num_columns): # iterate through every column in the image
#     bright_px_count = 0 # keep track of how many pixels in the column have a "high intensity" pixel (defined as being greater than the threshold value)
#     for row in range(0, num_rows):
#         if image1_grayscale[row, column] > thresh_val:
#             bright_px_count += 1
#     bright_pixels.append(bright_px_count)

# # Find the column with the most bright pixels, and choose this to be the slice we take:
# num_most_bright = 0
# index_most_bright = 0
# for x in range(0, len(bright_pixels)):
#     if bright_pixels[x] > num_most_bright:
#         num_most_bright = bright_pixels[x]
#         index_most_bright = x
# ydata = image1_grayscale[:, index_most_bright] # This is the data we'll use for the fitting

# # Would also like to find the (approximate) center of the peak within the slice:
# brightest_in_slice = 0
# index_brightest_in_slice = 0
# for x in range(0, len(ydata)):
#     if ydata[x] >= brightest_in_slice:
#         brightest_in_slice = ydata[x]
#         index_brightest_in_slice = x

# # Showing that we found the right part of the image:
# plt.imshow(image1_grayscale[index_brightest_in_slice-100:index_brightest_in_slice+100, (index_most_bright - 30):(index_most_bright + 30)])
# plt.show()


# # ============== Fitting the data from the image slice to a Gaussian function: ========================
# # Using curve_fit from scipy.optimize.

# # FIRST, FOR THE GAUSSIAN THE WAY I DEFINED IT:
# xdata = np.linspace(0, num_rows-1, num_rows)

# # Initial guesses for the values of the parameters:
# coeff_init = max_pix_val
# mean_init = index_brightest_in_slice
# sd_init = 1
# const_init = min_pix_val

# initial_guesses = np.array([coeff_init, mean_init, sd_init, const_init]) # Initial guesses for the parameters

# popt, pcov = curve_fit(gaussian, xdata, ydata, initial_guesses) # Run the fitting
# coeff, mean, sd, const = popt # these are the optimal paramters for the Gaussian function

# # NEXT, FOR THE GAUSSIAN THE WAY IT IS DEFINED IN STECK:
# xdata = np.linspace(0, num_rows-1, num_rows)

# # Initial guesses for the values of the parameters:
# coeff_init_steck = max_pix_val
# mean_init_steck = index_brightest_in_slice
# beam_rad_init_steck = 1
# const_init_steck = min_pix_val

# initial_guesses_steck = np.array([coeff_init_steck, mean_init_steck, beam_rad_init_steck, const_init_steck])

# popt_steck, pcov_steck = curve_fit(gaussian_steck, xdata, ydata, initial_guesses_steck) # Run the fitting
# coeff_steck, mean_steck, beam_rad_steck, const_steck = popt_steck # these are the optimal paramters for the Gaussian function for the Steck gaussian

# # ================== Generating data using the Gaussian fit and plotting it against the measured data
# # MY GAUSSIAN:
# f_vals = np.zeros(len(xdata)) # start with an array of zeros
# for i in range(0, len(xdata)):
#     f_vals[i] = gaussian(xdata[i], coeff, mean, sd, const)
# # f_vals now stores the values of the fitted Gaussian evaulated at each pixel

# # STECK GAUSSIAN:
# f_vals_steck = np.zeros(len(xdata)) # start with an array of zeros
# for i in range(0, len(xdata)):
#     f_vals_steck[i] = gaussian_steck(xdata[i], coeff_steck, mean_steck, beam_rad_steck, const_steck)
    
# # Plotting the fitted Gaussian against the original data from the camera
# # Over the entire column:
# plt.plot(xdata, ydata, 'b')
# plt.plot(xdata, f_vals, 'r')
# plt.plot(xdata, f_vals_steck, 'g')
# plt.title(f'Camera data in column with with index {index_most_bright}')
# plt.show()

# # Zoomed in plot:
# plt.plot(xdata[index_brightest_in_slice-8*int(sd):index_brightest_in_slice+8*int(sd)], ydata[index_brightest_in_slice-8*int(sd):index_brightest_in_slice+8*int(sd)], 'b')
# plt.plot(xdata[index_brightest_in_slice-8*int(sd):index_brightest_in_slice+8*int(sd)], f_vals[index_brightest_in_slice-8*int(sd):index_brightest_in_slice+8*int(sd)], 'r')
# plt.plot(xdata[index_brightest_in_slice-8*int(sd):index_brightest_in_slice+8*int(sd)], f_vals_steck[index_brightest_in_slice-8*int(sd):index_brightest_in_slice+8*int(sd)], 'g--')
# plt.title(f'(Zoomed) Camera data in column with index {index_most_bright}')
# plt.show()


# # ======== Calculate and output the 1/e^2 beam width using the Gaussian fit ===========

# # FIRST, FOR THE GAUSSIAN DEFINED IN TERMS OF ITS STANDARD DEVIATION:
# print('=== For the Gaussian defined in terms of its Standard Deviation: ===')
# # Note: I don't think that there is a simple way to convert from standard deviation to 1/e^2 beam radius, so I'll do it manually

# # The maximum value of the gaussian function is:
# max_value_gaussian = max(f_vals)
# # Correcting for constant:
# max_value_corr = max_value_gaussian - const
# # 1/e^2 of this value, then correcting for the constant, is:
# one_e2_value = max_value_corr * (1 / (np.e)**2) + const

# # Try to find the two horizontal coordinate values for which the value of the Gaussian is closest to one_e2_value:
# index_f_closest = 0
# index_f_second_closest = 1
# for i in range(0, len(f_vals)):
#     if np.abs(one_e2_value - f_vals[i]) < np.abs(f_vals[index_f_closest] - one_e2_value):
#         index_f_second_closest = index_f_closest
#         index_f_closest = i
#     elif np.abs(one_e2_value - f_vals[i]) < np.abs(f_vals[index_f_second_closest] - one_e2_value):
#         index_f_second_closest = i

# print(f'The max value of the Gaussian fit to the data is {max_value_gaussian}')
# print(f'Minus the constant term: = {max_value_corr}')
# print(f'1/e^2 of this value, plus the constant again, is {one_e2_value}')
# print(f'The two pixel values at which the Gaussian corresponds most closely to these values are px = {xdata[index_f_closest]} and px = {xdata[index_f_second_closest]}')
# print(f'These are a distance of abs({xdata[index_f_closest]} - {xdata[index_f_second_closest]}) =  {abs(xdata[index_f_closest] - xdata[index_f_second_closest])} px apart, which corresponds to a 1/e^2 beam width of: \n BEAM WIDTH = {abs(xdata[index_f_closest] - xdata[index_f_second_closest]) * 5.3} um')



# # SECOND, FOR THE GAUSSIAN DEFINED IN TERMS OF THE 1/e^2 beam radius:
# # This should be as simple as just doubling the fitted value for the 1/e^2 beam radius:
# print(' \n \n=== For the Gaussian defined in terms of its beam radius (Steck): ===')
# print(f'The fitted value of the beam radius is w = {beam_rad_steck} px')
# print(f'So the 1/e^2 beam width is twice this, at {2*beam_rad_steck} px')
# print(f'As a distance (using 5.3 um pixel size), this is" \n BEAM WIDTH = {5.3 * 2 * beam_rad_steck} um')

# # For context:
# print('\n \nThe fibre has an MFD of 5.5 +-1 um @ 850 nm, and we have a (150/13)*(150/75) = 23 X magnification system')
# print(f'This means that we should expect the 1/e^2 beam radius to be between {4.5 * 23} um and {6.5 * 23} um')