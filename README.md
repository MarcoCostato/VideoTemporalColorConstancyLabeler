# Video Temporal Color Constancy Labeler

A semi-automatic pipeline for the creation of Video Temporal Color Constancy Datasets.

## How to use

### Prerequisites:

The pipeline works by leveraging a few assumptions on the input data. Briefly:\

1. Each frame is affected by a single illuminant (weak assumption, the more uniform the illuminant in each frame the more precise is the label);
2. While the illuminant can change during the sequence, the change should be as smooth as possible;
3. At least two frames in a recorded sequence contain a fully visible full size [Macbeth ColorChecker](https://en.wikipedia.org/wiki/ColorChecker) (values for frames with missing targets will be interpolated)

The following preparation steps are necessary for the pipeline to work as intended in it's current version.\
To use the pipeline, the input data should have already been converted to single frames, demosaiced and saved in a easily processable format such as PNG. No other preprocessing steps are necessary nor suggested.\
Each video should be represented by it's own folder (naming is not important). Inside each video folder, all the frames should be saved inside a folder named **"RGB_png"**. Each frame in a video sequence should be ordered and named using the **"frame_XXXXXX"** convention.
All video folders should be located inside a root folder that will be used for processing (referred to as <path_to_base_directory>)

In order for the GUIs to work properly, a lot of images have to be saved on the device between scripts. As a consequence, a lot of memory is needed for the pipeline to run smoothly. Make sure to have AT LEAST 2 to 3 times the size of the original data available. Note that this requirement is heavily dependant on the number of "Personal Data" types you are looking obfuscate and could be much larger as the number grows.

### Step 0: Installing Dependencies

1. Clone the repository
2. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```
   Note: doing this will install all the dependencies needed to run all the code provided in the repo. The code was developed and tested using Python 3.10.11 . It is suggested to use the same version to avoid compatibility issues.

### Step 1: Label Extraction and Processing

1. Run:
   ```
   python ./Labeling/rawPNGChartLabeler.py <path_to_base_directory>
   ```
   to extract all the Ground Truths (**GTs**) for all frames where a target is present;
2. Run:
   ```
   python ./Labeling/outlierRemoval.py <path_to_base_directory>
   ```
   to detect and remove possible wrong detections and labels.
3. (Suggested) Run:
   ```
   python ./LabelingGUI/main.py
   ```
   to open the labeling GUI. Using the GUI, select the root folder and check for any obviously wrong detections in the various video folders. Any wrong detection can be deleted (suggested only if the frame is close to two correct detections) or corrected using the GUI. It is important to make sure that as little errors are present as possible, since they would be amplified in the next step
4. Run:
   ```
   python slerpInterpolation.py <path_to_base_directory>
   ```
   to interpolate missing data.

### Step 2: Target Obfuscation

Now that all labels have been extracted, it is suggested to run the following scripts to save the color target location in each frame.

1. Run:
   ```
   python ./Labeling/targetObfuscation.py <path_to_base_directory>
   ```
   to detect and save the target position
2. (Suggested) Run:
   ```
   python ./ObfuscationGUI/main.py
   ```
   to open the target obfuscation GUI. It works similarly to the previous GUI. Select the root folder, navigate the various video folders and check for any missed or wrong detections. They can be corrected by clicking on the frame with the wrong detection and manually correcting / adding the bounding box.

### Step 3: Personal Data Obfuscation

If you intend to make your dataset public, it is highly suggested that any kind of Personal Data that might be present in a scene (faces, license plates, documents, etc.) is first properly anonymized. This pipeline offers a simple script that can help with that, provided that you have the weight for a YOLOv8 model that detect the Personal Data of interest. Note that this might be more or less necessary depending on the types of scenes featured in the dataset (e.g. in an urban environment there will be a lot of faces and license plates, there will be a lot less in a forest setting).

1. Run:
   ```
   detection.py <base_directory> <yolo_model_path> <detection_label>
   ```
   where <yolo_model_path> is the path to the YOLOv8 model PyTorch weights and <detection_label> is the name of the feature that the model detects.\
   NOTE: if you are detecting **faces** or **license plates** use the "**faces**" and "**plates**" labels in order for the next step to work.
2. (If you detected faces or license plates AND followed the previous instructions) Run:
   ```
   python ./FaceDetectionGUI/main.py
   ```
   or
   ```
   python ./PlateDetectionGUI/main.py
   ```
   to open the respective GUIs and use them to check for any wrong/missed detections

### Step 4: Dataset Preparation

1. Run:
   ```
   python ./Labeling/datasetPreparation.py <input_base_dir> <output_base_dir>
   ```
   to finalize and generate the final dataset.
