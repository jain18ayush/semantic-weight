import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class MonoToPolySemanticsDataset(Dataset):
    """
    Dataset for the mono-to-poly semantics experiment.

    Phase 1: Classification based only on orientation (vertical vs. horizontal lines)
    Phase 2: Classification based on orientation AND color combinations
    """

    def __init__(self,
                 num_samples=5000,
                 img_size=32,
                 phase=1,
                 save_dir='dataset_images',
                 seed=42):
        """
        Initialize the dataset.

        Args:
            num_samples (int): Number of samples to generate
            img_size (int): Size of the square images
            phase (int): Training phase (1 or 2)
            save_dir (str): Directory to save generated images (if None, images are not saved)
            seed (int): Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.phase = phase
        self.seed = seed
        self.save_dir = save_dir

        # Set random seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create directory to save images if specified
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate the dataset
        self.data, self.labels, self.metadata = self._generate_dataset()

    def _generate_line_image(self, orientation, color):
        """
        Generate an image with a line of specified orientation and color.

        Args:
            orientation (str): 'vertical' or 'horizontal'
            color (str): 'red' or 'blue'

        Returns:
            numpy.ndarray: RGB image as a numpy array
        """
        # Create blank image (white background)
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)

        # Get color values
        if color == 'red':
            color_val = np.array([1.0, 0.0, 0.0])  # RGB for red
        else:  # blue
            color_val = np.array([0.0, 0.0, 1.0])  # RGB for blue

        # Draw line
        line_width = max(1, self.img_size // 8)  # Line is 1/8 of the image size
        center = self.img_size // 2

        if orientation == 'vertical':
            start_x = center - line_width // 2
            end_x = center + line_width // 2
            for x in range(start_x, end_x):
                img[0:self.img_size, x] = color_val
        else:  # horizontal
            start_y = center - line_width // 2
            end_y = center + line_width // 2
            for y in range(start_y, end_y):
                img[y, 0:self.img_size] = color_val

        # Add some noise to make it more challenging
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)

        return img

    def _generate_dataset(self):
        """
        Generate the full dataset based on the phase.

        Returns:
            tuple: (data, labels, metadata)
                data: torch tensor of shape (num_samples, 3, img_size, img_size)
                labels: torch tensor of shape (num_samples, 2)
                metadata: list of dicts with 'orientation' and 'color' for each sample
        """
        data = []
        labels = []
        metadata = []

        orientations = ['vertical', 'horizontal']
        colors = ['red', 'blue']

        for i in range(self.num_samples):
            # Randomly select orientation and color
            orientation = orientations[np.random.randint(0, 2)]
            color = colors[np.random.randint(0, 2)]

            # Generate image
            img = self._generate_line_image(orientation, color)

            # Convert to tensor format (C, H, W)
            img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)

            # Determine label based on phase
            if self.phase == 1:
                # Phase 1: Only orientation matters
                label = torch.tensor([1.0, 0.0] if orientation == 'vertical' else [0.0, 1.0],
                                   dtype=torch.float32)
            else:  # Phase 2
                # Phase 2: Combination of orientation and color
                is_class1 = (orientation == 'vertical' and color == 'red') or \
                            (orientation == 'horizontal' and color == 'blue')
                label = torch.tensor([1.0, 0.0] if is_class1 else [0.0, 1.0],
                                   dtype=torch.float32)

            data.append(img_tensor)
            labels.append(label)
            metadata.append({'orientation': orientation, 'color': color})

            # Save image if directory is specified
            if self.save_dir:
                img_save = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_save)
                img_name = f"{i}_{orientation}_{color}.png"
                img_pil.save(os.path.join(self.save_dir, img_name))

        # Stack all data and labels
        data = torch.stack(data)
        labels = torch.stack(labels)

        return data, labels, metadata

    def __len__(self):
        """Return the size of the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.data[idx], self.labels[idx]

    def get_metadata(self, idx):
        """Get metadata for a sample."""
        return self.metadata[idx]

    def switch_to_phase(self, phase):
        """Switch the dataset to a different phase."""
        if phase != self.phase:
            self.phase = phase
            # Regenerate labels based on new phase
            new_labels = []
            for meta in self.metadata:
                orientation = meta['orientation']
                color = meta['color']

                if phase == 1:
                    # Phase 1: Only orientation matters
                    label = torch.tensor([1.0, 0.0] if orientation == 'vertical' else [0.0, 1.0],
                                       dtype=torch.float32)
                else:  # Phase 2
                    # Phase 2: Combination of orientation and color
                    is_class1 = (orientation == 'vertical' and color == 'red') or \
                                (orientation == 'horizontal' and color == 'blue')
                    label = torch.tensor([1.0, 0.0] if is_class1 else [0.0, 1.0],
                                       dtype=torch.float32)

                new_labels.append(label)

            self.labels = torch.stack(new_labels)
            print(f"Dataset switched to Phase {phase}")

    def visualize_samples(self, num_samples=5):
        """Visualize random samples from the dataset with their labels."""
        indices = np.random.choice(len(self), num_samples, replace=False)

        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))

        for i, idx in enumerate(indices):
            img = self.data[idx].numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
            label = self.labels[idx].numpy()
            meta = self.metadata[idx]

            axes[i].imshow(img)

            if self.phase == 1:
                label_text = f"Orientation: {meta['orientation']}\nLabel: {'Vertical' if label[0] > 0.5 else 'Horizontal'}"
            else:
                is_class1 = label[0] > 0.5
                label_text = f"O: {meta['orientation']}, C: {meta['color']}\nLabel: {'Class 1' if is_class1 else 'Class 2'}"

            axes[i].set_title(label_text, fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()
        return fig