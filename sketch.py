import cv2
import numpy as np
import turtle
import os
import time
import tkinter
import _tkinter

def select_image():
    supported_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir('.') if f.lower().endswith(supported_extensions)]
    
    if not images:
        print("Error: No image files found in the folder!")
        return None
    
    if len(images) == 1:
        print(f"Found image: '{images[0]}'")
        return images[0]
    
    print("\n--- Available Sketches ---")
    for i, img_name in enumerate(images, 1):
        print(f"{i}. {img_name}")
    
    choice = input("\nEnter the number of the image you want to sketch: ")
    try:
        index = int(choice) - 1
        if 0 <= index < len(images):
            return images[index]
    except ValueError:
        pass
        
    print("Invalid choice. Selecting the first image by default.")
    return images[0]

IMAGE_PATH = select_image()

# Animation Controls (Smooth & Slower speed)
UPDATE_EVERY = 1      
DRAW_DELAY = 0.003    
PEN_SIZE = 1.0

def build_contours_and_fills(img):
    _, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    shapes = []
    EPSILON = 0.8

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 3:
            continue

        approx = cv2.approxPolyDP(cnt, EPSILON, True)
        if len(approx) < 3:
            continue

        pts = [pt[0] for pt in approx]
        parent_idx = hierarchy[i][3]
        
        depth = 0
        p = parent_idx
        while p != -1:
            depth += 1
            p = hierarchy[p][3]

        fill_color = "black" if depth % 2 == 0 else "white"

        shapes.append({
            'points': pts,
            'fill': fill_color,
            'is_solid': area > 30,
            'area': area
        })

    shapes.sort(key=lambda s: s['area'], reverse=True)
    return shapes

def main():
    if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
        print("Error: Could not load image.")
        return

    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image could not be loaded.")
        return

    height, width = img.shape
    max_dim = 650
    scale = max_dim / float(max(height, width))
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    img = cv2.resize(img, (new_width, new_height))
    height, width = new_width, new_height

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    shapes = build_contours_and_fills(blurred)

    screen = turtle.Screen()
    screen.setup(width + 100, height + 100)
    screen.title(f"Live Sketching — {IMAGE_PATH}")
    screen.bgcolor("white")

    turtle_drawer = turtle.Turtle()
    turtle_drawer.speed(0)
    turtle_drawer.pensize(PEN_SIZE)
    turtle_drawer.hideturtle()

    screen.tracer(0)

    offset_x = -width // 2
    offset_y = height // 2

    print(f"Starting live drawing of '{IMAGE_PATH}'...")
    start_time = time.time()
    current_point = 0

    try:
        for shape in shapes:
            points = shape['points']
            fill_color = shape['fill']
            is_solid = shape['is_solid']

            turtle_drawer.pencolor("black")
            turtle_drawer.fillcolor(fill_color)

            start_pt = points[0]
            turtle_drawer.penup()
            turtle_drawer.goto(start_pt[0] + offset_x, offset_y - start_pt[1])
            turtle_drawer.pendown()

            if is_solid:
                turtle_drawer.begin_fill()

            for point in points[1:]:
                x, y = point
                turtle_drawer.goto(x + offset_x, offset_y - y)

                current_point += 1
                if current_point % UPDATE_EVERY == 0:
                    screen.update()
                    if DRAW_DELAY > 0:
                        time.sleep(DRAW_DELAY)

            turtle_drawer.goto(start_pt[0] + offset_x, offset_y - start_pt[1])

            if is_solid:
                turtle_drawer.end_fill()

    except (turtle.Terminator, tkinter.TclError, _tkinter.TclError, Exception):
        print("\nDrawing window closed. Exiting...")
        return

    screen.update()
    elapsed = time.time() - start_time
    print(f"Drawing complete! Total points: {current_point}, Time taken: {elapsed:.2f} seconds.")
    
    try:
        screen.mainloop()
    except (turtle.Terminator, tkinter.TclError, _tkinter.TclError, Exception):
        pass

if __name__ == "__main__":
    main()