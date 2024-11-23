from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import torch
import pandas as pd

# Get current script directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Automatically find the weight file in the script directory
weight_file = None
for file in os.listdir(script_directory):
    if file.endswith('.pt'):
        weight_file = os.path.join(script_directory, file)
        break

if weight_file is None:
    raise FileNotFoundError("No weight file (.pt) found in the script directory.")

# Get image directory
images_directory = input("Enter the path to the image directory (e.g., /Users/images): ")

# Load trained model
model = YOLO(weight_file)

# Ask the user if they want to generate annotated images
generate_annotated_images = input("Do you want to generate annotated images? (yes/no): ").lower() == "yes"

# Create directory for annotated images if required
if generate_annotated_images:
    annotated_images_directory = os.path.join(script_directory, "annotated_images")
    os.makedirs(annotated_images_directory, exist_ok=True)

# Function to plot bounding boxes
def plot_bboxes(image, boxes, labels=[], colors=[], score=True, conf=None):
    if isinstance(image, Image.Image):
        # Convert PIL image to numpy array in RGB
        image = np.array(image)
    
    # Convert RGB to BGR for OpenCV processing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if not labels:
        labels = {0: 'Leukocyte', 1: 'Cornified', 2: 'Nucleated'}
    if not colors:
        colors = [(255, 0, 0), (0, 100, 0), (67, 161, 255)]
    
    for box in boxes:
        if score:
            label = labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%"
        else:
            label = labels[int(box[-1])]
        if conf and box[-2] <= conf:
            continue
        color = colors[int(box[-1])]
        box_label(image, box, label, color)
    
    # Convert back to RGB before returning
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.002), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = 1
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 6, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 6, txt_color, thickness=tf, lineType=cv2.LINE_AA)

# Function to detect estrous stage
def determine_estrus_stage(boxes, names):
    class_indices = boxes.data[:, -1].cpu().to(dtype=torch.int32).numpy()
    total_cells = len(class_indices)
    leukocytes_count = sum(names[i] == "Leukocyte" for i in class_indices)
    cornified_count = sum(names[i] == "Cornified" for i in class_indices)
    nucleated_count = sum(names[i] == "Nucleated" for i in class_indices)

    # Calculate percentages
    leukocytes_percentage = (leukocytes_count / total_cells) * 100 if total_cells else 0
    cornified_percentage = (cornified_count / total_cells) * 100 if total_cells else 0
    nucleated_percentage = (nucleated_count / total_cells) * 100 if total_cells else 0

    # Determine if low cell count
    is_low_cell_count = '***' if total_cells <= 35 else ''

    # Determine estrous stage based on cell counts and percentages
    if total_cells == 0:
        predicted_stage = "No cells detected"
    elif total_cells > 35:
        if cornified_percentage >= 80 or (cornified_percentage >= 70 and leukocytes_percentage <= 10):
            predicted_stage = "Estrus"
        elif leukocytes_percentage >= 75 or (leukocytes_percentage >= 60 and cornified_percentage <= 10):
            predicted_stage = "Diestrus"
        elif nucleated_percentage >= 35 or nucleated_percentage >= 70:
            predicted_stage = "Proestrus"
        else:
            predicted_stage = "Metestrus"
    else:  # Low cell count (<=35)
        if leukocytes_count < 5:
            if cornified_count > nucleated_count:
                predicted_stage = "Estrus"
            elif nucleated_count > cornified_count:
                predicted_stage = "Proestrus"
            elif nucleated_count == cornified_count == 0 and leukocytes_count >= 1:
                predicted_stage = "Diestrus"
            else:
                predicted_stage = "Metestrus"
        elif leukocytes_count >= 5:
            if leukocytes_percentage >= 70:
                predicted_stage = "Diestrus"
            elif nucleated_count >= 1 or cornified_count >= 1:
                predicted_stage = "Metestrus"
            else:
                predicted_stage = "Metestrus"

    return predicted_stage, total_cells, leukocytes_percentage, cornified_percentage, nucleated_percentage, is_low_cell_count

results = []

# Loop through each image in the directory and process
for image_name in sorted(os.listdir(images_directory)):  # Added sorting here for file listing
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(images_directory, image_name)
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_np_bgr = image_np[..., ::-1]

        # Updated model prediction call with specified parameters
        result_generator = model.predict(image_np_bgr, conf=0.25, max_det=1000, stream=True)
        for result in result_generator:
            # Determine estrous stage and percentages
            stage_classification, total_cells, leuko_perc, corni_perc, nucl_perc, is_low_cell_count = determine_estrus_stage(result.boxes.data, model.names)
            print(f"Processed {image_name}, Estrous cycle stage: {stage_classification}, "
                  f"Total Cells: {total_cells}, Leukocytes: {leuko_perc:.2f}%, Cornified: {corni_perc:.2f}%, Nucleated: {nucl_perc:.2f}%")

            # Collect results
            results.append({
                "image_name": image_name,
                "estrous_stage": stage_classification,
                "total_cells": total_cells,
                "leukocytes_percentage": leuko_perc,
                "cornified_percentage": corni_perc,
                "nucleated_percentage": nucl_perc,
                "is_low_cell_count": is_low_cell_count
            })

            if generate_annotated_images:
                # Plot bounding boxes with the specified confidence threshold
                annotated_image = plot_bboxes(image, result.boxes.data, conf=0.25)
                if annotated_image.size == 0:
                    print(f"Warning: Annotated image for {image_name} is empty. Skipping saving.")
                    continue
                # Save the annotated image to the annotated_images directory
                save_path = os.path.join(annotated_images_directory, f"annotated_{image_name}")
                cv2.imwrite(save_path, cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                print(f"Annotated image saved to: {save_path}")

# Save the results after processing all images
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('image_name')  # Sort the DataFrame by 'image_name' before saving

# Generate a unique results filename
output_csv_path = os.path.join(script_directory, 'results.csv')
counter = 1
while os.path.exists(output_csv_path):
    output_csv_path = os.path.join(script_directory, f'results{counter}.csv')
    counter += 1

df_results.to_csv(output_csv_path, index=False)
print(f"Results saved to {output_csv_path}")
