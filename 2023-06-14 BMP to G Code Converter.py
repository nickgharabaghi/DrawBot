import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

# Image to load in:
filename = r"C:\Users\nickg\Desktop\Side Projects\DrawBot\DrawBot\UnoptimizedPath.bmp"

# Parameters for the printer:
start_x = 0 # mm - starting x position
start_y = 0 # mm - starting y position
speed = 100 # mm/s - how fast to move the printer head

# Load in the bitmap file that we want to turn into G-code
def load_in_image(filename):
    '''Takes a raw image file of a maze, and returns a cleaned-up version.'''
    # Load in image:
    maze_colour = mpimg.imread(filename)
    # Show the inputted image, to make sure it's what you want:
    print("Inputted image:")
    maze_colour_plot = plt.imshow(maze_colour) 
    plt.show()

load_in_image(filename)

# Create a file for the G-code
with open(r"C:\Users\nickg\Desktop\Side Projects\DrawBot\DrawBot\maze_g_code.gcode", "w") as file:
	# Create header:
    file.write("G28 \n") # home all axes
    file.write(f"G0 F{speed} X{start_x} Y{start_y}\n") # move to start location
    file.write("M4 \n") # start spindle (lower pen)

    # Create full instructions to solve the maze:
    file.write("ayyyy \n")

    # Footer:
    file.write("M5 \n") # stop spindle (raise pen)
    file.write("M84 X Y \n") # disable the stepper motors



