import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tifffile
from torchvision.transforms import RandomAffine


# Set environment variable for compatibility with display systems
os.environ['QT_QPA_PLATFORM'] = 'xcb'

class AdversarialPatchGenerator:
    def __init__(self, target_class=11, patch_ratio=0.2, grid_size=3):
        """Initialize the AdversarialPatchGenerator.

        Args:
            target_class (int): The class ID of the target object to attack (default: 11).
            patch_ratio (float): Ratio determining patch size relative to bounding box (default: 0.2).
            grid_size (int): Size of the grid for patch position search (default: 3).
        """
        # Determine device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load and configure the YOLO model
        self.model = YOLO('yolov8n_TrafficSigns.pt').to(self.device)
        self.model.eval()
        
        # Initialize parameters
        self.target_class = target_class
        self.patch_ratio = patch_ratio
        self.grid_size = grid_size
        self.loss_history = []
        self.confidence_threshold = 0.25
        self.use_adaptive_learning = True

    def apply_transforms(self,image_tensor):
        """Apply random affine transformations to ensure robustness.
        
        Args:
            image_tensor (torch.Tensor): Input image tensor of shape [C, H, W]
            
        Returns:
            torch.Tensor: Transformed image tensor
        """    
        # Define random transformation parameters
        transform = RandomAffine(
            degrees=15,           # ±15° rotation
            translate=(0.1, 0.1), # ±10% positional shift
            scale=(0.9, 1.1)      # 90-110% scaling
        )
        
        # Apply transformation (requires NCHW format)
        if len(image_tensor.shape) == 3:  # CHW format
            transformed = transform(image_tensor.unsqueeze(0)).squeeze(0)
        else:  # Already has batch dimension
            transformed = transform(image_tensor)
            
        return transformed
    def total_variation_loss(self,patch):
        """Calculate Total Variation loss to encourage patch smoothness.
        
        Args:
            patch (torch.Tensor): Patch tensor of shape [C, H, W]
            
        Returns:
            torch.Tensor: TV loss value
        """
        # Calculate differences in x and y directions
        diff_x = torch.abs(patch[:, :, :-1] - patch[:, :, 1:])
        diff_y = torch.abs(patch[:, :-1, :] - patch[:, 1:, :])
        
        # Sum all differences
        tv_loss = torch.sum(diff_x) + torch.sum(diff_y)
        
        return tv_loss
    def apply_perspective(self, image_tensor):
        # Convert tensor to NumPy array, detaching gradients
        img_np = image_tensor.detach().permute(1, 2, 0).cpu().numpy()
        
        # Ensure image is uint8 for OpenCV if needed
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Get image dimensions
        h, w = img_np.shape[:2]
        
        # Define source points as float32
        src_pts = np.float32([
            [0, 0],      # Top-left
            [w-1, 0],    # Top-right
            [w-1, h-1],  # Bottom-right
            [0, h-1]     # Bottom-left
        ])
        
        # Compute destination points and ensure float32
        dst_pts = src_pts + np.random.uniform(low=-50, high=50, size=(4, 2))
        dst_pts = dst_pts.astype(np.float32)  # Cast to float32 to satisfy OpenCV
        
        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply warp
        warped = cv2.warpPerspective(img_np, M, (w, h))
        
        # Convert back to tensor
        warped_tensor = torch.from_numpy(warped).permute(2, 0, 1).float().to(self.device) / 255.0
        
        return warped_tensor
    def evaluate_robustness(self, image_tensor, patches_with_positions, bbox, iterations=20):
        """Evaluate patch robustness under various physical-world transformations.
        
        Args:
            image_tensor (torch.Tensor): Clean image tensor.
            patches_with_positions (list): List of (patch, grid_position) tuples.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            iterations (int): Number of random transformations to test.
            
        Returns:
            dict: Evaluation results with success rates.
        """
        import torchvision.transforms.functional as TF
        import numpy as np
        
        print("\n=== Physical Robustness Evaluation ===")
        
        # Define transformation parameters
        angles = [-15, -10, -5, 0, 5, 10, 15]  # Rotation angles
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]     # Scaling factors
        brightness = [0.8, 0.9, 1.0, 1.1, 1.2]  # Brightness factors
        
        # Tracking metrics
        total_tests = 0
        success_count = 0
        confidence_values = []
        
        # Test fixed combinations of transformations
        print("\nTesting fixed transformations:")
        for angle in angles:
            for scale in [0.9, 1.0, 1.1]:  # Reduced set for fixed tests
                total_tests += 1
                
                # Apply base patches
                perturbed = self.apply_multiple_patches(image_tensor, patches_with_positions, bbox)
                
                # Apply transformations
                transformed = perturbed.clone()
                # Ensure the angle is a float
                a = float(angle)
                transformed = TF.rotate(transformed, a)
                
                # Adjust scale with interpolation
                if scale != 1.0:
                    h, w = transformed.shape[1:]
                    new_h, new_w = int(h * scale), int(w * scale)
                    transformed = TF.resize(transformed, [new_h, new_w])
                    transformed = TF.center_crop(transformed, [h, w])
                
                # Run through model
                with torch.no_grad():
                    results = self.model(transformed.unsqueeze(0))
                    _, conf = self.calculate_disappearance_loss(results, self.target_class)
                    confidence_values.append(conf)
                    
                    success = conf < self.confidence_threshold
                    if success:
                        success_count += 1
                    
                    print(f"  Angle: {a:+3.1f}°, Scale: {scale:.1f}, "
                        f"Conf: {conf:.4f}, Success: {success}")
        
        # Random combinations
        print("\nTesting random transformations:")
        for i in range(iterations):
            # Randomly sample transformations
            angle = int(np.random.choice(angles))
            scale = np.random.choice(scales)
            bright = np.random.choice(brightness)
            perspective = np.random.random() < 0.5
            
            total_tests += 1
            
            # Apply base patches
            perturbed = self.apply_multiple_patches(image_tensor, patches_with_positions, bbox)
            
            # Apply transformations
            transformed = perturbed.clone()
            transformed = TF.adjust_brightness(transformed, bright)
            transformed = TF.rotate(transformed, float(angle))
            
            # Adjust scale
            if scale != 1.0:
                h, w = transformed.shape[1:]
                new_h, new_w = int(h * scale), int(w * scale)
                transformed = TF.resize(transformed, [new_h, new_w])
                transformed = TF.center_crop(transformed, [h, w])
            
            # Apply perspective if selected
            if perspective:
                transformed = self.apply_perspective(transformed)
            
            # Run through model
            with torch.no_grad():
                results = self.model(transformed.unsqueeze(0))
                _, conf = self.calculate_disappearance_loss(results, self.target_class)
                confidence_values.append(conf)
                
                success = conf < self.confidence_threshold
                if success:
                    success_count += 1
                
                print(f"  Test {i+1}: Angle: {angle:+3d}°, Scale: {scale:.1f}, "
                    f"Brightness: {bright:.1f}, Perspective: {perspective}, "
                    f"Conf: {conf:.4f}, Success: {success}")
        
        # Calculate statistics
        success_rate = success_count / total_tests * 100
        avg_conf = sum(confidence_values) / len(confidence_values)
        
        results = {
            "success_rate": success_rate,
            "average_confidence": avg_conf,
            "success_count": success_count,
            "total_tests": total_tests,
        }
        
        print(f"\nResults Summary:")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Average Confidence: {avg_conf:.4f}")
        print(f"  Successful Attacks: {success_count}/{total_tests}")
        
        return results

    def load_image(self, image_path):
        """Load and preprocess an image.

            Args:
                image_path (str): Path to the image file.

            Returns:
                torch.Tensor: Preprocessed image tensor.
        """
            # Read image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
            # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to 640x640 (YOLO input size)
        resized_img = cv2.resize(image, (640, 640))
        # Convert to tensor, permute to (C, H, W), normalize to [0, 1]
        image_tensor = torch.tensor(resized_img, dtype=torch.float32).permute(2, 0, 1).to(self.device) / 255.0
        return image_tensor

    def get_bounding_box(self, image_tensor):
        """Detect the bounding box of the target class in the image and shrink it to fit the object.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            tuple: (x1, y1, x2, y2) coordinates of the adjusted bounding box.
        """
        with torch.no_grad():
            results = self.model(image_tensor.unsqueeze(0))
            for i in range(len(results[0].boxes)):
                if int(results[0].boxes.cls[i].item()) == self.target_class:
                    bbox = results[0].boxes.xyxy[i].clone().cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    # Shrink the bounding box to better fit the stop sign
                    shrink_factor = 0.8  # Adjust as needed
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    new_width = width * shrink_factor
                    new_height = height * shrink_factor
                    x1 = int(center_x - new_width / 2)
                    x2 = int(center_x + new_width / 2)
                    y1 = int(center_y - new_height / 2)
                    y2 = int(center_y + new_height / 2)
                    return x1, y1, x2, y2
        raise ValueError(f"Target class {self.target_class} not found in image")

    def apply_patch(self, image_tensor, patch, bbox, grid_position=None):
        """Apply a single patch to the image at a specified position.

            Args:
                image_tensor (torch.Tensor): Input image tensor.
            patch (torch.Tensor): Patch tensor to apply.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            grid_position (tuple, optional): (row, col) position in the grid.

        Returns:
            torch.Tensor: Image tensor with the patch applied.
        """
        x1, y1, x2, y2 = bbox
        H, W = image_tensor.shape[1:]
        result_tensor = image_tensor.clone()
        
        # Calculate patch size based on bounding box
        obj_width = x2 - x1
        obj_height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        patch_height, patch_width = patch.shape[1], patch.shape[2]        
        # Determine patch position
        if grid_position is not None:
            row, col = grid_position
            cell_width = obj_width / self.grid_size
            cell_height = obj_height / self.grid_size
            rel_x = (col + 0.5) / self.grid_size
            rel_y = (row + 0.5) / self.grid_size
            patch_center_x = x1 + obj_width * rel_x
            patch_center_y = y1 + obj_height * rel_y
        else:
            patch_center_x = center_x
            patch_center_y = center_y
        
        # Calculate patch boundaries
        patch_x1 = int(patch_center_x - patch_width / 2)
        patch_y1 = int(patch_center_y - patch_height / 2)
        patch_x2 = patch_x1 + patch_width
        patch_y2 = patch_y1 + patch_height
        
        # Ensure patch stays within image boundaries
        valid_y1 = max(0, patch_y1)
        valid_y2 = min(H, patch_y2)
        valid_x1 = max(0, patch_x1)
        valid_x2 = min(W, patch_x2)
        
        patch_valid_y1 = max(0, valid_y1 - patch_y1)
        patch_valid_y2 = patch_height - max(0, patch_y2 - valid_y2) if patch_y2 > H else patch_height
        patch_valid_x1 = max(0, valid_x1 - patch_x1)
        patch_valid_x2 = patch_width - max(0, patch_x2 - valid_x2) if patch_x2 > W else patch_width
        
        # Apply patch if valid region exists
        if valid_y2 > valid_y1 and valid_x2 > valid_x1:
            result_tensor[:, valid_y1:valid_y2, valid_x1:valid_x2] = patch[
                :, patch_valid_y1:patch_valid_y2, patch_valid_x1:patch_valid_x2
            ]
        
        return result_tensor

    def apply_multiple_patches(self, image_tensor, patches_with_positions, bbox):
        """Apply multiple patches to the image.

        Args:
            image_tensor (torch.Tensor): Input image tensor.
            patches_with_positions (list): List of (patch, grid_position) tuples.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).

        Returns:
            torch.Tensor: Image tensor with all patches applied.
        """
        result_tensor = image_tensor.clone()
        for patch, grid_position in patches_with_positions:
            result_tensor = self.apply_patch(result_tensor, patch, bbox, grid_position)
        return result_tensor

    def calculate_disappearance_loss(self, results, target_class):
        """Calculate the loss based on the confidence of the target class.

        Args:
            results: Detection results from the YOLO model.
            target_class (int): Target class ID.

        Returns:
            tuple: (loss tensor, maximum confidence value).
        """
        max_conf = 0.0
        found_target = False
        target_confidences = []
        
        # Extract confidence scores for target class
        for i in range(len(results[0].boxes)):
            cls = int(results[0].boxes.cls[i].item())
            if cls == target_class:
                conf = results[0].boxes.conf[i]
                target_confidences.append(conf)
                found_target = True
        
        if found_target and target_confidences:
            confidences_tensor = torch.stack(target_confidences)
            max_conf_tensor = torch.max(confidences_tensor)
            max_conf = max_conf_tensor.item()
            loss = max_conf_tensor
        else:
            max_conf = 0.0
            # Small positive loss if target is not detected (attack success)
            loss = torch.tensor(0.01, device=self.device, requires_grad=True)
        
        return loss, max_conf

    def load_patch_image(self, image_path, patch_width,patch_height):
        """Load and preprocess a patch image.

        Args:
            image_path (str): Path to the patch image file.
            patch_size (int): Size to resize the patch to.

        Returns:
            torch.Tensor: Preprocessed patch tensor.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Failed to load image {image_path}, using random initialization")
                return torch.rand((3, patch_width, patch_height), device=self.device)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (patch_width, patch_height))
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(self.device) / 255.0
            print(f"Loaded custom patch image from {image_path}")
            return img_tensor
        except Exception as e:
            print(f"Error loading patch image: {e}, using random initialization")
            return torch.rand((3, patch_size, patch_size), device=self.device)
    def train_patch_at_position(self, image_tensor, bbox, grid_position, epochs, base_lr=0.1, init_patch=None, existing_patches=None,aspect_ratio=1):
        """Train a patch at a specific grid position.

        Args:
            image_tensor (torch.Tensor): Input image tensor.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            grid_position (tuple): (row, col) position in the grid.
            epochs (int): Number of training epochs.
            base_lr (float): Base learning rate (default: 0.1).
            init_patch (torch.Tensor, optional): Initial patch tensor.
            existing_patches (list, optional): List of previously optimized (patch, position) tuples.

        Returns:
            tuple: (optimized patch tensor, best confidence value).
        """
        # Handle default value for existing_patches
        if existing_patches is None:
            existing_patches = []

        # Calculate patch size based on bounding box width and patch ratio
        patch_size = int((bbox[2] - bbox[0]) * np.sqrt(self.patch_ratio))

        patch_width = patch_size*aspect_ratio

        # Initialize patch
        if init_patch is not None:
            patch = init_patch.clone()
        else:
            patch = torch.rand((3, patch_width, patch_size), device=self.device)
        patch.requires_grad = True

        # Initialize best patch and confidence
        best_patch = patch.clone().detach()
        best_conf = 1.0
        stagnation_counter = 0

        # Set up optimizer
        optimizer = torch.optim.Adam([patch], lr=base_lr)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Apply existing patches and the current patch to create the base perturbed image
            base_image = self.apply_multiple_patches(image_tensor, existing_patches, bbox)
            perturbed = self.apply_patch(base_image, patch, bbox, grid_position)

            # Always compute the untransformed image's detection results
            results_untrans = self.model(perturbed.unsqueeze(0))
            loss_untrans, conf_untrans = self.calculate_disappearance_loss(results_untrans, self.target_class)

            # Randomly decide whether to apply transformations (80% chance)
            if np.random.random() < 0.8:
                perturbed_trans = perturbed.clone()
                if np.random.random() < 0.5:
                    perturbed_trans = self.apply_transforms(perturbed_trans)  # Rotation, scale, translation
                if np.random.random() < 0.5:
                    perturbed_trans = self.apply_perspective(perturbed_trans)  # Perspective warping
                results_trans = self.model(perturbed_trans.unsqueeze(0))
                loss_trans, conf_trans = self.calculate_disappearance_loss(results_trans, self.target_class)
            else:
                # If no transformations, use the untransformed results
                loss_trans = loss_untrans
                conf_trans = conf_untrans

            # Compute total loss with total variation regularization
            tv_weight = 0.1  # Adjustable hyperparameter
            tv_loss = self.total_variation_loss(patch)
            total_loss = loss_trans + tv_weight * tv_loss

            # Backpropagate and optimize
            if total_loss.requires_grad:
                total_loss.backward()
                optimizer.step()

            # Clamp patch values to [0, 1]
            with torch.no_grad():
                patch.clamp_(0, 1)

            # Adaptive learning rate adjustment based on untransformed confidence
            if self.use_adaptive_learning and epoch > 0 and epoch % 5 == 0:
                for param_group in optimizer.param_groups:
                    if conf_untrans < 0.3 and conf_untrans > 0.05:
                        param_group['lr'] = base_lr * 0.5
                    elif conf_untrans <= 0.05:
                        param_group['lr'] = base_lr * 0.1

            # Update best patch based on untransformed confidence
            if conf_untrans < best_conf:
                best_conf = conf_untrans
                best_patch = patch.clone().detach()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Early stopping if untransformed confidence is sufficiently low
            if conf_untrans < 0.2:
                print(f"  Early stopping at epoch {epoch}: untrans conf {conf_untrans:.4f}")
                break

            # Handle stagnation based on untransformed confidence
            if stagnation_counter >= 10:
                if conf_untrans > 0.5:
                    with torch.no_grad():
                        noise = torch.randn_like(patch) * 0.2
                        patch.add_(noise).clamp_(0, 1)
                    stagnation_counter = 0
                    print(f"  Adding noise at epoch {epoch}")
                elif stagnation_counter >= 15:
                    print(f"  Stopping at epoch {epoch} due to stagnation, best untrans conf: {best_conf:.4f}")
                    break

            # Log progress periodically
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Grid {grid_position} | Epoch {epoch}/{epochs} | Trans Conf: {conf_trans:.4f} | Untrans Conf: {conf_untrans:.4f} | Best Untrans: {best_conf:.4f}")

        return best_patch, best_conf
    def train(self, image_path, num_patches=1, epochs_per_location=30, base_lr=0.1, refinement_epochs=50, patch_image_path=None,aspect_ratio=1):
        """Train multiple adversarial patches on the image with visualization.

            Args:
                image_path (str): Path to the target image.
                num_patches (int): Number of patches to generate (default: 1).
            epochs_per_location (int): Epochs for initial grid search (default: 30).
            base_lr (float): Base learning rate (default: 0.1).
            refinement_epochs (int): Epochs for patch refinement (default: 50).
            patch_image_path (str, optional): Path to initial patch image.

        Returns:
            tuple: (list of (patch, position) tuples, final confidence).
        """
        # Load and preprocess image
        image_tensor = self.load_image(image_path)
        x1, y1, x2, y2 = self.get_bounding_box(image_tensor)
        bbox = (x1, y1, x2, y2)
        
        patch_size = int((x2 - x1) * np.sqrt(self.patch_ratio))
        patch_width=patch_size*aspect_ratio
        # Load initial patch if provided
        init_patch = None
        if patch_image_path is not None:
            init_patch = self.load_patch_image(patch_image_path, patch_width=patch_width,patch_height=patch_size)
        
        optimized_patches = []
        
        # Initialize visualization
        plt.ion()  # Enable interactive mode
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        location_results = np.ones((self.grid_size, self.grid_size)) * 1.0
        cbar = None
        # Optimize each patch
        for patch_idx in range(num_patches):
            print(f"\n=== Optimizing Patch {patch_idx + 1}/{num_patches} ===")
            best_conf = 1.0
            best_location = None
            best_patch = None
            
            # Grid search for best patch position
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    grid_position = (row, col)
                    print(f"\nTesting grid position {grid_position} for patch {patch_idx + 1}")
                    
                    patch, conf = self.train_patch_at_position(
                        image_tensor,
                        bbox,
                        grid_position,
                        epochs_per_location,
                        base_lr,
                        init_patch=init_patch,
                        existing_patches=optimized_patches
                    )
                    
                    location_results[row, col] = conf
                    
                    if conf < best_conf:
                        best_conf = conf
                        best_location = grid_position
                        best_patch = patch
                    
                    # Update visualization
                    # Heatmap of confidences
                    ax3.clear()
                    im = ax3.imshow(location_results, cmap='viridis_r')

                    # Add colorbar only once, then update it
                    if cbar is None:
                        cbar = plt.colorbar(im, ax=ax3)
                    else:
                        cbar.update_normal(im)  # Update existing colorbar with new data

                    ax3.set_title(f"Confidence Heatmap for Patch {patch_idx + 1}")
                    # Add text annotations to heatmap
                    for r in range(self.grid_size):
                        for c in range(self.grid_size):
                            ax3.text(c, r, f"{location_results[r, c]:.2f}",
                                    ha="center", va="center",
                                    color="white" if location_results[r, c] > 0.5 else "black")

                    # Update current perturbed image (assume apply_multiple_patches exists)
                    current_perturbed = self.apply_multiple_patches(
                        image_tensor, optimized_patches + [(patch, grid_position)], bbox
                    )
                    display_img = current_perturbed.permute(1, 2, 0).detach().cpu().numpy()
                    ax1.clear()
                    ax1.imshow(display_img)
                    ax1.set_title(f"Current: Grid {grid_position}\nConf: {conf:.4f}")

                    # Update current patch
                    patch_img = patch.detach().cpu().permute(1, 2, 0).numpy()
                    ax2.clear()
                    ax2.imshow(patch_img)
                    ax2.set_title(f"Patch for Grid {grid_position}")

                    plt.tight_layout()
                    plt.pause(0.01)  # Brief pause to update display
                    plt.draw()
            
            # Refine the best patch found
            print(f"\nRefining patch at position {best_location}")
            refined_patch, refined_conf = self.train_patch_at_position(
                image_tensor,
                bbox,
                best_location,
                refinement_epochs,
                base_lr * 0.5,
                init_patch=best_patch,
                existing_patches=optimized_patches
            )
            
            if refined_conf < best_conf:
                best_conf = refined_conf
                best_patch = refined_patch
            
            optimized_patches.append((best_patch, best_location))
        
        # Evaluate final result
        final_perturbed = self.apply_multiple_patches(image_tensor, optimized_patches, bbox)
        with torch.no_grad():
            results = self.model(final_perturbed.unsqueeze(0))
            _, final_conf = self.calculate_disappearance_loss(results, self.target_class)
        
        print(f"\nFinal confidence with {num_patches} patches: {final_conf:.4f}")
        
        # Save patches and final image
        for i, (patch, location) in enumerate(optimized_patches):
            patch_img = (patch.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            tifffile.imwrite(f"patch_{i+1}_grid{location}.tiff", patch_img)
        
        final_perturbed_img = (final_perturbed.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        tifffile.imwrite("final_perturbed_multi.tiff", final_perturbed_img)
        
        # Display final visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(final_perturbed_img / 255)
        plt.title(f"Final Image with {num_patches} Patches\nConfidence: {final_conf:.4f}")
        plt.axis('off')
        plt.savefig("final_result_multi.png")
        plt.show()

        print("\nEvaluating physical robustness...")
#         robustness_results = self.evaluate_robustness(
#     image_tensor, optimized_patches, bbox, iterations=20
# )
        
        return optimized_patches, final_conf

if __name__ == "__main__":
    # Example usage
    generator = AdversarialPatchGenerator(target_class=14, patch_ratio=0.05, grid_size=10)
    optimized_patches, final_conf = generator.train(
        image_path="stop.jpeg",
        num_patches=2,
        epochs_per_location=20,
        base_lr=0.01,
        refinement_epochs=20,
        patch_image_path="patch.jpeg",
        aspect_ratio=2
    )