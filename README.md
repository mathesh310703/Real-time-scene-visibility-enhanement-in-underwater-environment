# Real-Time Scene Visibility Enhancement in Underwater Environment

This project develops an advanced real-time underwater scene visibility
enhancement system that utilizes a 3-level deep multi-patch hierarchical neural network to effectively tackle underwater scene degradation. Engineered for high-speed processing, the system ensures real-time performance while addressing the complexities of underwater environments. By incorporating a multi-patch hierarchical approach, the model enhances both local and global image features, restoring visual clarity and improving scene perception across various underwater applications.

## How to Run
### Prerequisites
- Python >= 3.10
- NumPy
  - Use a version < 2.0.0
- OpenCV
- PyTorch
- Pillow

### Usage
1. Clone this repository.
3. `$ python main.py`

### Training
1. Ensure that all frames under a single video are stored in the same directory.
 - Refer the SUVE dataset for example.
2. Edit the following parameters inside the `train.py` file as required.
```py
LEARNING_RATE = 
EPOCHS =
GPU =
BATCH_SIZE = 
```
3. Train the model using `$ python train.py`