**License:** [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)

# Kazakh_Handwritten_Letter_Dataset

Kazakh language has 42 letters (from which 33 are native to Russian as well). This dataset consists of 22,662 samples of handwritten Kazakh letters collected from 108 participants. Each letter appears approximately 540 times across the dataset.

The dataset is split into training, validation, and test sets with 18,047 (79.6%), 2,310 (10.2%), and 2,310 (10.2%) images respectively, assigned at the sheet level to prevent writer-style leakage across splits.

## Dataset Statistics

- **Total images:** 22,662
- **Classes:** 42 Kazakh letters
- **Image format:** 128×128 grayscale PNG
- **Participants:** 108
- **Split method:** Sheet-level (80/10/10 train/val/test)
- **Per-class distribution:** ~430 train, ~55 val, ~55 test per letter

## Methodology

### Data Collection

108 participants each filled one sheet of paper. Each sheet contained a 21×12 table where participants wrote each of the 42 Kazakh letters 5 times. No writer identity information was retained.

### Digitization

Sheets were scanned using the iPhone 15 Notes application (grayscale mode) and saved as PDFs. All PDFs were converted to PNG format at 300 DPI using the `pdf2image` library.

### Processing Pipeline

1. **Letter Extraction:** Automated grid-based cropping identified individual letter cells from each scanned sheet.
2. **Artifact Removal:** Border-line remnants from the grid (detected via connected-component analysis) were removed to prevent preprocessing noise from affecting model training.
3. **Normalization:** All images were resized to 128×128 pixels via padding-to-square (to avoid distortion) followed by scaling. This preserves fine stroke detail better than smaller resolutions.

### Quality Control

Contact sheets were generated for visual inspection of all letters. A small number of edge cases (2.3% of images) where ink touched the frame border were flagged for manual review.

### Train/Val/Test Split

The split was assigned at the **sheet level**, not the image level. This ensures that all samples from a given writer remain in a single split, preventing writer-style characteristics from leaking across train/validation/test sets and inflating reported accuracy. A reproducible 80/10/10 split was achieved using a fixed random seed (42).

## Dataset Structure

dataset_split_v2/

├── train/

│======├── а/

│======├── б/

│======├── ... (42 letter folders)

├── val/

│======├── а/

│======├── ...

└── test/

│======├── а/

│======├── ...

└── ...


metadata.csv  # filepath, label, sheet_id, split

## Usage

### Loading with PyTorch

```python
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.ImageFolder("dataset_split_v2/train", transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
```

### Loading with Metadata CSV

```python
import csv
import torch
from PIL import Image

metadata = list(csv.DictReader(open("metadata.csv")))
train_rows = [row for row in metadata if row['split'] == 'train']

for row in train_rows:
    img = Image.open(row['filepath']).convert('L')
    label = row['label']
```

## Baseline Results

A ResNet18 model pre-trained on ImageNet and fine-tuned on this dataset achieved:

- **Test Accuracy:** 91.84%
- **Macro Average F1:** 0.919

Per-class results and confusion matrix are available in `baseline_results/`.

Most letters achieved 90%+ accuracy. Letters with slightly lower accuracy (и, ң, ұ, ч, ц) have similar visual characteristics and are naturally more challenging to distinguish.

## Known Limitations

- **Grid-line artifacts:** A small subset (~2.3%) of training images contain faint grid-line remnants from the scanning process, though these were largely mitigated via v2 normalization.
- **Handwriting variation:** High variability in letter size, slant, and stroke pressure across writers. This is natural but does introduce classification ambiguity for visually similar letters.
- **Class imbalance:** Minor—most letters have 540 samples, but a few have 530-539 due to occasional cropping failures during digitization.

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). You are free to share, adapt, and use this dataset for any purpose, provided you give appropriate credit.

## Citation

If you use this dataset in your work, please cite it as:

```bibtex
@dataset{kazakh_handwritten_letters_2026,
  title={Kazakh Handwritten Letter Dataset},
  author={Sheriyeva Nuray M.},
  year={2026},
  url={https://github.com/[Nuray-Sheriyeva]/kzdtst}
}
```

## Contact & Feedback

For questions, issues, or feedback about this dataset, please open an issue on GitHub or contact nuray.sheriyeva@gmail.com.
