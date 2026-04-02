# Color Chart Localization
Official implementation of the code from [Robust camera-independent color chart localization using YOLO](https://www.sciencedirect.com/science/article/pii/S0167865525001138?via%3Dihub)


## Installation
1. Clone the repository.
2. Install the required dependencies by running:
    ```
    pip install -r requirements.txt
    ```

## Running the annotate.py script
1. Navigate to the project directory.
2. Run the script by executing:
    ```
    python annotate.py <path/to/image> {--out-file <path/to/output_file>} {--segment_background} --device <cpu|cuda> {--get_viz}
    ```
    Example:
    ```
    python annotate.py demo/1_8D5U5524.png --segment_background --device cuda --get_viz

    ```

## Cite
If you use this code in your research, please cite our paper:
```
@article{cogo2025robust,
  title={Robust camera-independent color chart localization using YOLO},
  author={Cogo, Luca and Buzzelli, Marco and Bianco, Simone and Schettini, Raimondo},
  journal={Pattern Recognition Letters},
  volume={192},
  pages={51--58},
  year={2025},
  publisher={Elsevier}
}
```
