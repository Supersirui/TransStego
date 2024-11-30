


# **TransStego**

TransStego is a project for performing image steganography using advanced techniques such as transformers and deep learning. The system hides data within images while preserving visual quality and robustness against various attacks. This project contains the code for paper titled "A Transformer-Based Adversarial Network Framework for Steganography".

---

## **Features**

- **Image Steganography:** Embed and extract hidden information in images.
- **High Visual Quality:** Maintains high PSNR and SSIM metrics for embedded images.
- **Robustness:** Resistant to various noise attacks and detection methods.


---

## **Installation**

### **Prerequisites**
- Python >= 3.8
- `pip` package manager
- A virtual environment tool (optional but recommended)

### **Setup**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Supersirui/TransStego.git
   cd TransStego
   ```
   
2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```


## **Usage**
1. **Training**

To train the model:

   ```bash
   python train.py
   ```
You can change the setting of the training process in "./cfg/setting."

2. **Encoding**

To hide data in an image:

   ```bash
   python demo.py <model_path> --image <input_image> --save_dir <output_dir> --secret <secret_message>
   ```
We provide a demo to test, you can use it by:

   ```bash
   python demo.py ./saved_models/masktrain/en_name.pth --images_dir ./images --save_dir <output_dir>
   ```

3. **Decoding**

To extract hidden data from an image:

   ```bash
   python decode_image.py --input <encoded_image>
   ```

## **Project Structure**

1. **plaintext**
```
TransStego/
├── cfg/               # Configuration files
├── saved_models/      # Pretrained models (tracked with Git LFS)
├── images/            # Sample images for testing
├── Output/            # Encoded images and residuals
├── dataset.py         # Create dataset
├── Encoder.py         # Encoder model
├── train.py           # Training script
├── demo.py            # Encoding script
├── decode_image.py    # Decoding script
├── model.py           # training process
├── transforms.py      # Transfom the image
├── utils.py
├── vit.py
├── requirements.txt   # Python dependencies
├── LICENSE            # License file
└── README.md          # Project documentation
```

## **Performance Metrics**

1. - **PSNR (Peak Signal-to-Noise Ratio)**: Achieves high PSNR for encoded images.
2. - **SSIM (Structural Similarity Index)**: Retains excellent structural quality.
3. - **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual similarity between original and encoded images.



## **License**
This project is licensed under the MIT License. See the LICENSE file for details.
