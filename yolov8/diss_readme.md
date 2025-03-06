

# Adversarial Patch Attack on YOLO Object Detection

## Overview

This project implements an adversarial attack against YOLO-based object detection models by generating optimized patches that reduce the confidence of detecting a specific target class. The attack works by strategically placing small patches within the object's bounding box to cause misclassification or disappearance.

## Attack Mechanism

The attack utilizes gradient-based optimization to craft adversarial patches that, when overlaid on an image, minimize the detection confidence of the target object. The approach includes:

1. **Target Class Identification**: The attack is focused on a specific class (default: `11`).
2. **Patch Generation**: Randomly initialized patches are optimized iteratively.
3. **Grid Search Optimization**: Patches are placed in different positions within the bounding box to find the most effective locations.
4. **Physical-World Transformations**: The attack accounts for real-world robustness by applying affine transformations, perspective warping, and brightness variations.
5. **Total Variation Regularization**: A smoothness constraint is applied to ensure the generated patches are visually plausible.

## Loss Function

The attack optimizes an adversarial loss function that consists of two key components:

### **1. Disappearance Loss (Main Loss)**

This loss function minimizes the confidence of the target class in the YOLO detection model.

#### **Mathematical Definition:**
The disappearance loss is formulated as:


$J_d(x,y) = \max_{s \in S^2, b \in B} P(s,b,y,f_\theta (x))$

where:
- \( f_\theta(x) \) represents the output of the YOLO object detector.
- \( P(s,b,y,f_\theta(x)) \) extracts the probability of the target class \( y \) in grid cell \( s \) and bounding box \( b \).
- The attack minimizes this probability until it falls below the detection threshold (typically 0.25).


✅ **Objective**: Reduce the confidence of the target class to make it disappear from the detection results.

### **2. Total Variation Loss (TV Loss)**

This loss is used as a regularization term to ensure the patch is smooth and does not contain noisy patterns that might be unrealistic.

#### **Mathematical Definition:**
The total variation loss is defined as:


$TV(M_x \cdot \delta) = \sum_{i,j} |(M_x \cdot \delta)_{i+1,j} - (M_x \cdot \delta)_{i,j}| + |(M_x \cdot \delta)_{i,j+1} - (M_x \cdot \delta)_{i,j}|$

where:
- \( M_x \) is the mask defining the patch area.
- \( \delta \) is the adversarial perturbation.
- The summation ensures smoothness by penalizing pixel-level differences.



✅ **Objective**: Prevent adversarial patches from having sharp edges or unrealistic pixel variations.



The final loss function combines the disappearance loss and total variation loss:

$\mathcal{L}_{\text{total}} = J_d(x,y) + \lambda TV(M_x \cdot \delta)$


where \( \lambda \) is a weight parameter to balance between adversarial effectiveness and smoothness.




✅ **Overall Objective:**

1. Minimize the detection confidence of the target class.
2. Ensure the patch remains smooth and natural-looking.

## Dependencies

Ensure you have the required dependencies installed:

```bash
pip install torch torchvision opencv-python matplotlib ultralytics tifffile
```

## How to Run

### 1. Training an Adversarial Patch

```bash
python Disappearance.py
```

This will generate adversarial patches for the target class and apply them to a given image.

### 2. Custom Parameters

You can modify key parameters within the script:

- **Target Class**: Change `target_class` in `AdversarialPatchGenerator`.
- **Patch Size Ratio**: Adjust `patch_ratio` to control patch size relative to the object.
- **Grid Size**: Modify `grid_size` to set the number of search positions for patches.

### 3. Example Execution

To generate patches and evaluate their impact:

```python
# Initialize attack
attack = AdversarialPatchGenerator(target_class=10, patch_ratio=0.02, grid_size=10)

# Run attack
optimized_patches, final_conf = attack.train(
    image_path="road666_png.rf.efa2191f5abb5266654b44edb18bb15a.jpg",
    num_patches=3,
    epochs_per_location=20,
    base_lr=0.1,
    refinement_epochs=20,
    patch_image_path="patch.jpeg"
)
```

## Results

- The attack generates optimized adversarial patches that minimize the detection confidence of the target object.
- The patches are saved as `.tiff` images, and the final adversarial image is visualized and saved as `final_result_multi.png`.


---


