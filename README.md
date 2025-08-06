## ODES (Object Detection for Estrous Staging)

**ODES** is a machine learning tool that analyzes **stained images** to detect and classify cell types for monitoring the estrous cycle in female mice.

> **NOTE:** ODES is designed to work with **stained images**. Unstained or poorly stained images may lead to inaccurate results.
---

## Instructional Video

For a step-by-step video guide on setting up and using ODES, watch the instructional video on YouTube:

[ODES Tutorial Video](https://www.youtube.com/watch?v=lUOiC_60pRg)


---

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Image Preparation](#image-preparation)
- [Uninstallation](#uninstallation)

---

## Installation

These instructions only need to be followed once:

1. **Install Anaconda**

   Download and install [Anaconda](https://www.anaconda.com/products/distribution).

2. **Download ODES Files**

   Clone this repository or download the following files from the ODES folder:
   - `ODES.py`
   - `finalweight.pt`

3. **Create ODES Folder**

   Create a new folder on your desktop named **ODES** and place the downloaded files into it.

4. **Set Up Conda Environment**

   Open Terminal (Mac/Linux) or Anaconda Prompt (Windows) and execute the following commands:
   ```bash
   conda create --name odes
   conda activate odes
   ```

5. **Install Dependencies**

   Install the required dependencies:
   ```bash
   conda install numpy pillow opencv pandas
   conda install -c pytorch pytorch
   pip install ultralytics

---

## Usage

Follow these steps each time you want to use ODES:

1. **Prepare Image Folder**

   Place your image folder in the **ODES** folder on your desktop.

2. **Activate Conda Environment**

   Open Terminal (Mac/Linux) or Anaconda Prompt (Windows) and activate the `odes` environment:
   ```bash
   conda activate odes
   ```

3. **Navigate to ODES Folder**

   Navigate to the ODES folder:
   ```bash
   cd /path/to/ODES
   ```
   Replace `/path/to/ODES` with the full path to the ODES folder on your desktop (e.g., `/Users/YourUsername/Desktop/ODES` or `C:\Users\YourUsername\Desktop\ODES`).

4. **Run the Script**

   Execute the script:
   ```bash
   python ODES.py
   ```

5. **Provide Image Folder Path**

   When prompted, enter the path to your image directory:
   - Example input: `testImages`
   - You can type just the folder name if the folder is inside the **ODES** folder.
   - For other locations, provide the full path (e.g., `/Users/YourUsername/images`).

6. **Choose Output Options**

   When prompted, decide whether to generate annotated images:
   - Type `yes` to generate annotated images and create:
     - A `results.csv` file with all classifications.
     - A new folder called `annotated_images` containing the annotated images.
   - Type `no` to only generate the `results.csv` file.

   Both output files will be saved in the **ODES** folder.

---

## Image Preparation

To achieve the best results with ODES, please ensure your images are prepared according to the following guidelines:

- **Objective Magnification:** 10X objective lens is recommended. The model was trained primarily on 10X images, so using this magnification aligns with the learned parameters and reduces variability in predictions.

- **Image Format:** ODES supports the following image formats: PNG, JPG, JPEG, and BMP. While all these formats are compatible, PNG is recommended for faster processing. 

- **Staining:** Ensure images are properly stained. Unstained images reduce the model's ability to accurately distinguish cell boundaries and characteristics, as it relies on these features to identify and classify cells effectively.

- **Cell Clumping:** Avoid images where cells are clumped together. ODES may struggle to classify cells accurately when they overlap or lack clear boundaries, which can complicate the detection of individual cell features. This issue can be exacerbated by biological and histological factors, such as sticky vaginal excretions, which may lead to cell clumping. ODES is optimized for images with clear and distinct cell features, similar to human judgment processes.

- **Cell Count:** Be aware that very low cell counts in images can make stage classification difficult.
---

## Uninstallation

To remove the ODES environment and its dependencies:

1. **Open Terminal or Anaconda Prompt**

   Launch Terminal (Mac/Linux) or Anaconda Prompt (Windows).

2. **Remove Conda Environment**

   Execute the following command:
   ```bash
   conda remove --name odes --all
   ```

3. **Confirm Uninstallation**

   Confirm the action when prompted.

4. **Delete ODES Folder**

   Optionally, delete the **ODES** folder from your desktop to remove the script files.
