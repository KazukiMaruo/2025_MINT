"""
TODOS:
    - 
Info:
    - Numerosity from 1 ~ 6
  
"""

# ~~~~~~~~~~~~~ Libraries
from PIL import Image, ImageDraw
import math
import random
import os
import json
# ~~~~~~~~~~~~~ Libraries ~~~~~~~~~~~~~



# ~~~~~~~~~~~~~ Functions
def compute_radius(width, screen_width_cm, distance_cm, visual_angle):
    """
    Compute the radius of a circle in pixels based on visual angle, screen dimensions, and viewing distance.  
    Parameters:
    - width: Screen width in pixels
    - screen_width_cm: Physical screen width in cm
    - distance_cm: Distance from the viewer to the screen in cm
    - visual_angle: Visual angle in degrees 
    Returns:
    - radius_pixels: Radius of the circle in pixels
    """
    visual_angle_radians = math.radians(visual_angle)   # Convert visual angle to radians
    object_size_cm = 2 * (distance_cm * math.tan(visual_angle_radians / 2))     # Calculate the size of the object in cm on the screen
    pixels_per_cm = width / screen_width_cm     # Convert the size to pixels based on screen width and resolution
    radius_pixels = (object_size_cm / 2) * pixels_per_cm    # Calculate the radius in pixels
    return radius_pixels



def euclidean_distance(x1, y1, x2, y2): # Function to calculate Euclidean distance between two points
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)



def control_singledotsize(width, height, circle_radius_pixels, n_trials):
    """
    Generates single dot sise controled stimuli with varying numbers of dots (from 1 ~ 6), ensuring no overlap and positioning
    them inside a circle.
    
    Parameters:
    - width: Width of the screen resolution (in pixels).
    - height: Height of the screen resolution (in pixels).
    - circle_radius_pixels: Radius of the circle where dots are placed (in pixels).
    """
    # Base directory for storing images
    base_dir = os.getcwd()  # Gets current directory of this script
    target_dir = os.path.join(base_dir, 'stimuli', 'visual', 'singledotsize_cont')
    
    # define dot radius_pixels
    dot_radius_pixels = int(circle_radius_pixels/4)

    # Loop through each numerosity (from 1 to 6)
    for numerosity in range(1, 7):
        # Create a directory for each numerosity if it doesn't exist
        numerosity_dir = os.path.join(target_dir, f"numerosity_{numerosity}")
        os.makedirs(numerosity_dir, exist_ok=True)

        # Generate 70 images for each numerosity
        for img_num in range(n_trials):
            # Create a blank image with gray background
            image = Image.new("RGB", (width, height), (128, 128, 128))
            draw = ImageDraw.Draw(image)

            # Calculate the center of the image
            center_x, center_y = width // 2, height // 2

            # Draw diagonal red lines (optional for visual guide)
            draw.line((0, 0, width, height), fill=(255, 0, 0), width=1)  # Top left to bottom right
            draw.line((width, 0, 0, height), fill=(255, 0, 0), width=1)  # Top right to bottom left

            # Store positions of the dots
            dot_positions = []

            # Place the first dot randomly inside the circle
            angle = random.uniform(0, 2 * math.pi)
            radius_offset = random.uniform(0, circle_radius_pixels)
            dot_x = center_x + radius_offset * math.cos(angle)
            dot_y = center_y + radius_offset * math.sin(angle)
            dot_positions.append((dot_x, dot_y))

            # Draw the first dot
            draw.ellipse(
                (dot_x - dot_radius_pixels, dot_y - dot_radius_pixels,
                 dot_x + dot_radius_pixels, dot_y + dot_radius_pixels),  # Small dot size
                fill="black"
            )

            # Place the remaining dots at a controlled distance from the first dot
            for _ in range(1, numerosity):
                while True:
                    # Random angle around the center
                    angle = random.uniform(0, 2 * math.pi)
                    radius_offset = random.uniform(0, circle_radius_pixels)
                    dot_x = center_x + radius_offset * math.cos(angle)
                    dot_y = center_y + radius_offset * math.sin(angle)

                    # Check for overlap and if it's within the circle
                    overlap = False
                    for (existing_x, existing_y) in dot_positions:
                        if euclidean_distance(dot_x, dot_y, existing_x, existing_y) < 2 * dot_radius_pixels + 3:
                            overlap = True
                            break

                    # If no overlap and inside the circle, place the dot
                    if not overlap and euclidean_distance(dot_x, dot_y, center_x, center_y) <= circle_radius_pixels:
                        dot_positions.append((dot_x, dot_y))
                        draw.ellipse(
                            (dot_x - dot_radius_pixels, dot_y - dot_radius_pixels,
                             dot_x + dot_radius_pixels, dot_y + dot_radius_pixels),
                            fill="black"
                        )
                        break  # Exit the while loop once a valid dot is placed

            # Save the image with a unique name
            image_path = os.path.join(numerosity_dir, f"image_{img_num+1}.png")
            image.save(image_path)
    print(f"When the singledotsize is controlled, single dot radius (px) from 1 to 6 is: {dot_radius_pixels}")
    return dot_radius_pixels



def control_totaldotsize(width, height, circle_radius_pixels, n_trials):
    """
    Generates total dot sise controled stimuli with varying numbers of dots (from 1 ~ 6), ensuring no overlap and positioning
    them inside a circle.
    
    Parameters:
    - width: Width of the screen resolution (in pixels).
    - height: Height of the screen resolution (in pixels).
    - circle_radius_pixels: Radius of the circle where dots are placed (in pixels).
    """
    # Base directory for storing images
    base_dir = os.getcwd()
    target_dir = os.path.join(base_dir, 'stimuli', 'visual', 'totaldotsize_cont')

    # Loop through each numerosity (from 1 to 6)
    dot_radius_pixels_list = []
    for numerosity in range(1, 7):
        # Create a directory for each numerosity if it doesn't exist
        numerosity_dir = os.path.join(target_dir, f"numerosity_{numerosity}")
        os.makedirs(numerosity_dir, exist_ok=True)

        # Calculate the total area for the dots and adjust dot radius
        total_area = math.pi * (circle_radius_pixels) ** 2  # Total area occupied by the dots
        single_dot_area = (total_area/3) / numerosity  # Area for each dot
        dot_radius_pixels = math.sqrt(single_dot_area / math.pi)  # Adjusted radius based on the area
        dot_radius_pixels_list.append(dot_radius_pixels)

        # Generate 70 images for each numerosity
        for img_num in range(n_trials):
            # Create a blank image with gray background
            image = Image.new("RGB", (width, height), (128, 128, 128))
            draw = ImageDraw.Draw(image)

            # Calculate the center of the image
            center_x, center_y = width // 2, height // 2

            # Draw diagonal red lines
            draw.line((0, 0, width, height), fill=(255, 0, 0), width=1)  # Line from top left to bottom right
            draw.line((width, 0, 0, height), fill=(255, 0, 0), width=1)  # Line from top right to bottom left

            # Store positions of the dots
            dot_positions = []

            # Place the first dot randomly inside the circle
            angle = random.uniform(0, 2 * math.pi)
            radius_offset = random.uniform(0, circle_radius_pixels)
            dot_x = center_x + radius_offset * math.cos(angle)
            dot_y = center_y + radius_offset * math.sin(angle)
            dot_positions.append((dot_x, dot_y))

            # Draw the first dot
            draw.ellipse(
                (dot_x - dot_radius_pixels, dot_y - dot_radius_pixels, 
                 dot_x + dot_radius_pixels, dot_y + dot_radius_pixels),  # Small dot size
                fill="black"
            )

            # Place the remaining dots at a controlled distance from the first dot
            for _ in range(1, numerosity):
                while True:
                    # Random angle around the first dot
                    angle = random.uniform(0, 2 * math.pi)
                    radius_offset = random.uniform(0, circle_radius_pixels)
                    # Set the dot position around the specified Euclidean distance from the first dot
                    dot_x = center_x + radius_offset * math.cos(angle)
                    dot_y = center_y + radius_offset * math.sin(angle)

                    # Check if the new dot overlaps with the first dot or goes out of bounds
                    overlap = False
                    for (existing_x, existing_y) in dot_positions:
                        if euclidean_distance(dot_x, dot_y, existing_x, existing_y) < 2 * dot_radius_pixels + 3:
                            overlap = True
                            break

                    # If no overlap and the dot is inside the circle, draw the dot
                    if not overlap and euclidean_distance(dot_x, dot_y, center_x, center_y) <= circle_radius_pixels:
                        dot_positions.append((dot_x, dot_y))
                        draw.ellipse(
                            (dot_x - dot_radius_pixels, dot_y - dot_radius_pixels, 
                             dot_x + dot_radius_pixels, dot_y + dot_radius_pixels),  # Small dot size
                            fill="black"
                        )
                        break  # Exit while loop after placing the dot

            # Save the image with a unique name
            image_path = os.path.join(numerosity_dir, f"image_{img_num+1}.png")
            image.save(image_path)
    print(f"When the totaldotsize is controlled, single dot radius (px) from 1 to 6 is: {dot_radius_pixels_list}")
    return dot_radius_pixels_list



def control_circumference(width, height, circle_radius_pixels, n_trials):
    """
    Generates totatl circumference controled stimuli with varying numbers of dots (from 1 ~ 6), ensuring no overlap and positioning
    them inside a circle.
    
    Parameters:
    - width: Width of the screen resolution (in pixels).
    - height: Height of the screen resolution (in pixels).
    - circle_radius_pixels: Radius of the circle where dots are placed (in pixels).
    """
    # Base directory for storing images
    base_dir = os.getcwd()
    target_dir = os.path.join(base_dir, 'stimuli', 'visual', 'circumference_cont')

    dot_radius_pixels_list = []

    # Loop through each numerosity (from 1 to 6)
    for numerosity in range(1, 7):
        # Create a directory for each numerosity if it doesn't exist
        numerosity_dir = os.path.join(target_dir, f"numerosity_{numerosity}")
        os.makedirs(numerosity_dir, exist_ok=True)

        # Calculate the total area for the dots and adjust dot radius
        if numerosity == 1:
            total_area = math.pi * (circle_radius_pixels) ** 2  # Total area of circle based on the visual angle
            single_dot_area = (total_area / 2) / numerosity  # Area for each dot
            dot_radius_pixels = math.sqrt(single_dot_area / math.pi)  # Adjusted radius based on the area
            circumference = 2 * math.pi * dot_radius_pixels
        
        else:
            circumference_per_dot = circumference / numerosity
            dot_radius_pixels = circumference_per_dot / (2 * math.pi)

        # Store the calculated dot radius
        dot_radius_pixels_list.append(dot_radius_pixels)

        # Generate 70 images for each numerosity
        for img_num in range(n_trials):
            # Create a blank image with gray background
            image = Image.new("RGB", (width, height), (128, 128, 128))
            draw = ImageDraw.Draw(image)

            # Calculate the center of the image
            center_x, center_y = width // 2, height // 2

            # Draw diagonal red lines
            draw.line((0, 0, width, height), fill=(255, 0, 0), width=1)  # Line from top left to bottom right
            draw.line((width, 0, 0, height), fill=(255, 0, 0), width=1)  # Line from top right to bottom left

            # Store positions of the dots
            dot_positions = []

            # Place the first dot randomly inside the circle
            angle = random.uniform(0, 2 * math.pi)
            radius_offset = random.uniform(0, circle_radius_pixels)
            dot_x = center_x + radius_offset * math.cos(angle)
            dot_y = center_y + radius_offset * math.sin(angle)
            dot_positions.append((dot_x, dot_y))

            # Draw the first dot
            draw.ellipse(
                (dot_x - dot_radius_pixels, dot_y - dot_radius_pixels,
                 dot_x + dot_radius_pixels, dot_y + dot_radius_pixels),  # Small dot size
                fill="black"
            )

            # Place the remaining dots at a controlled distance from the first dot
            for _ in range(1, numerosity):
                while True:
                    # Random angle around the first dot
                    angle = random.uniform(0, 2 * math.pi)
                    radius_offset = random.uniform(0, circle_radius_pixels)
                    # Set the dot position around the specified Euclidean distance from the first dot
                    dot_x = center_x + radius_offset * math.cos(angle)
                    dot_y = center_y + radius_offset * math.sin(angle)

                    # Check if the new dot overlaps with the first dot or goes out of bounds
                    overlap = False
                    for (existing_x, existing_y) in dot_positions:
                        if euclidean_distance(dot_x, dot_y, existing_x, existing_y) < 2 * dot_radius_pixels + 3:
                            overlap = True
                            break

                    # If no overlap and the dot is inside the circle, draw the dot
                    if not overlap and euclidean_distance(dot_x, dot_y, center_x, center_y) <= circle_radius_pixels:
                        dot_positions.append((dot_x, dot_y))
                        draw.ellipse(
                            (dot_x - dot_radius_pixels, dot_y - dot_radius_pixels,
                             dot_x + dot_radius_pixels, dot_y + dot_radius_pixels),  # Small dot size
                            fill="black"
                        )
                        break  # Exit while loop after placing the dot

            # Save the image with a unique name
            image_path = os.path.join(numerosity_dir, f"image_{img_num+1}.png")
            image.save(image_path)
    print(f"When the circumference is controlled, single dot radius (px) from 1 to 6 is: {dot_radius_pixels_list}")
    return dot_radius_pixels_list



def background_image(width, height):
    # Base directory for storing images
    base_dir = os.getcwd()
    target_dir = os.path.join(base_dir, 'stimuli', 'visual')
    os.makedirs(target_dir, exist_ok=True)
    image = Image.new("RGB", (width, height), (128, 128, 128))
    draw = ImageDraw.Draw(image)

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Draw diagonal red lines (optional for visual guide)
    draw.line((0, 0, width, height), fill=(255, 0, 0), width=1)  # Top left to bottom right
    draw.line((width, 0, 0, height), fill=(255, 0, 0), width=1)  # Top right to bottom left

    # Save the image with a unique name
    image_path = os.path.join(target_dir, "background.png")
    image.save(image_path)
# ~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~



# ~~~~~~~~~~~~~ Parameters
width, height = 1920, 1080  # screen pixel dimensions
distance_cm = 50  # Distance from the viewer in cm
screen_width_cm = 50  # Screen width in cm
visual_angle = 2 # visual angle
circle_radius_pixels = compute_radius(width, screen_width_cm, distance_cm, visual_angle) # compute the raidus of circle within the specified visual angle
# ~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~


# ~~~~~~~~~~~~~ Constant
n_trials = 70 # e.g., 70 trials for numerosity 3 in singledot controlled condition
n_condition = 3
total_n_trials = n_trials * n_condition
numerosity = [1, 2, 3, 4, 5, 6]
# ~~~~~~~~~~~~~ Constant ~~~~~~~~~~~~~


# ~~~~~~~~~~~~~ Generator
background_image(width, height)
singledotsize = control_singledotsize(width, height, circle_radius_pixels, n_trials)
totaldotsize = control_totaldotsize(width, height, circle_radius_pixels, n_trials)
circumference = control_circumference(width, height, circle_radius_pixels, n_trials) # this function generates the images and retunrs the radius of each dot
# ~~~~~~~~~~~~~ Generator ~~~~~~~~~~~~~


# ~~~~~~~~~~~~~ Output the stimuli data into json
base_dir = os.getcwd()
target_dir = os.path.join(base_dir, 'stimuli', 'visual')
file_name = 'param.json'
file_path = os.path.join(target_dir, file_name)
data = {
    'singledotsize_cont_radius_px': {str(i+1): singledotsize for i in range(6)},
    'totaldotsize_cont_radius_px': {str(i+1): totaldotsize[i] for i in range(6)},
    'circumference_cont_radius_px': {str(i+1): circumference[i] for i in range(6)},
    'n_trials': n_trials,
    'n_condition': n_condition,
    'total_n_trials': total_n_trials,
    'numerosity': numerosity
}
# Create and write the JSON file
with open(file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)
print(f"JSON file created successfully at: {file_path}")
# ~~~~~~~~~~~~~ Output the stimuli data into json ~~~~~~~~~~~~~
