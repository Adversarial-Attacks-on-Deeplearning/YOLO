# Creation Attack Loss 
## Overview 
The creation attack loss mechanism aims to fool the model into recognizing nonexistent objects. Similar to the 
“adversarial patch” approach, the goal is to create a physical sticker that can be added to any existing scene. 
Contrary to the adversarial patch, rather than causing a misclassification, the aim is to create a new classification 
(i.e., a new object detection) where none existed before.

The objective proposed in this attack uses a composite loss function that first aims at creating a new object 
localization, followed by a targeted “mis-classification.” However, since we are working with YOLOv8, the objective is 
to maximize the target class probability until the target class probability exceeds 50%.

let fθ (x) represent the full output tensor of YOLO v8 on input
scene x, and let P(s, b, y, fθ (x)) represent the probabil-
ity assigned to class y in box b of grid cell s. 
loss is then

Jc (x, y) = 1 - P(s, b, y, fθ (x))
 
## Attack mechanism

The attack utilizes gradient-based optimization to craft adversarial patches that, when overlaid on an image, maximize the detection confidence of the target object.
The approach includes:

1. **Target Class Identification**: The target class to be added.
2. **Patch Generation**: Use initial patch that optimized iteratively.
3. **Get patch location**: Patch is placed at the grid that has the maximum propability for the target class.
4. **Apply patch**: The patch size is modified to fit the grid.
5. **Patch optimization**: Use gradient-based optimization to the patch to decrease loss.

## results
![img0](https://github.com/user-attachments/assets/abbbf8b7-7cda-4da8-ae1f-c9be9738056b)


![det0](https://github.com/user-attachments/assets/f41df6ae-13ee-4544-a617-3e68ac1ceb32)


![img3](https://github.com/user-attachments/assets/8be3b681-4bf5-4417-8de1-70a77f9a9a33)


![det3](https://github.com/user-attachments/assets/b77fe74d-e09f-441a-8963-04a363969c2c)



### Initial patch used:

![ss_patch](https://github.com/user-attachments/assets/d19615d9-c82d-4cc2-a99a-c5eeb8dab2d1)

