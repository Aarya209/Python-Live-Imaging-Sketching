import cv2
import numpy as np
import turtle
import os
import time
# Path to the image file saved in the project folder
IMAGE_PATH = "sketch_image.png"

# Animation Controls
UPDATE_EVERY = 2      # Refresh screen every N points (Lower = smoother/slower live drawing)
DRAW_DELAY = 0.001    # Delay per point in seconds (increase for slower human-like drawing)
PEN_SIZE = 1.5
PEN_COLOR = "black"

def get_depth(idx, hierarchy):
    """Calculate nesting depth of a contour using hierarchy."""
    depth = 0
    curr = hierarchy[0][idx][3]  # Parent index
    while curr != -1:
        depth += 1
        curr = hierarchy[0][curr][3]
    return depth

def build_contours(contours, hierarchy):
    """Process contours with polygon approximation and depth mapping."""
    if hierarchy is None:
        return []

    result = []
    EPSILON = 0.8  # Precision for polygon approximation

    for i, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, EPSILON, True)
        if len(approx) < 3:
            continue

        depth = get_depth(i, hierarchy)
        result.append({
            'approx': approx,
            'depth': depth,
            'points': [pt[0] for pt in approx]
        })

    # Sort shapes: Outer shapes/larger depth drawn systematically
    result.sort(key=lambda item: item['depth'])
    return result

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Could not find '{IMAGE_PATH}'. Please make sure the image is in the same folder as this script!")
        return

    # Load image in grayscale format
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image could not be loaded. Check file integrity.")
        return

    # Resize image to fit comfortably on screen
    height, width = img.shape
    max_dim = 650
    scale = max_dim / float(max(height, width))
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    img = cv2.resize(img, (new_width, new_height))
    height, width = new_width, new_height

    # Apply Gaussian Blur to smooth noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Extract edges using Canny edge detector
    edges = cv2.Canny(blurred, threshold1=80, threshold2=180)

    # Find contours and full nesting hierarchy (RETR_TREE)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    shapes = build_contours(contours, hierarchy)

    # Setup Turtle Screen
    screen = turtle.Screen()
    screen.setup(width + 120, height + 120)
    screen.title("Maa Durga — Slow Live Sketching")
    screen.bgcolor("white")

    # Setup Turtle Drawer Pen
    turtle_drawer = turtle.Turtle()
    turtle_drawer.speed(0)
    turtle_drawer.pensize(PEN_SIZE)
    turtle_drawer.hideturtle()

    # Turn off automatic screen updates for custom refresh rate
    screen.tracer(0)

    offset_x = -width // 2
    offset_y = height // 2

    print("Starting slow live drawing...")
    start_time = time.time()

    total_points = sum(len(shape['points']) for shape in shapes)
    current_point = 0

    for shape in shapes:
        points = shape['points']
        depth = shape['depth']

        if len(points) < 3:
            continue

        # Alternate black / white fill or outline based on nesting level depth
        if depth % 2 == 0:
            fill_color = PEN_COLOR
        else:
            fill_color = "white"

        turtle_drawer.pencolor(PEN_COLOR)
        turtle_drawer.fillcolor(fill_color)

        # Move to starting point of shape
        start_pt = points[0]
        turtle_drawer.penup()
        turtle_drawer.goto(start_pt[0] + offset_x, offset_y - start_pt[1])
        turtle_drawer.pendown()

        turtle_drawer.begin_fill()

        # Trace through all points sequentially
        for point in points[1:]:
            x, y = point
            turtle_drawer.goto(x + offset_x, offset_y - y)

            current_point += 1

            # Update screen periodically for slow realistic sketching effect
            if current_point % UPDATE_EVERY == 0:
                screen.update()
                if DRAW_DELAY > 0:
                    time.sleep(DRAW_DELAY)

        # Close the shape contour and fill region
        turtle_drawer.goto(start_pt[0] + offset_x, offset_y - start_pt[1])
        turtle_drawer.end_fill()

    screen.update()
    elapsed = time.time() - start_time
    print(f"Drawing complete! Total points: {current_point}, Time taken: {elapsed:.2f} seconds.")
    
    screen.mainloop()

if __name__ == "__main__":
    main()