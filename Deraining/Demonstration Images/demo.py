import os
import sys
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as TF

# Add the parent directory 
sys.path.append('..')
from MPRNet import MPRNet

def center_crop(img, max_width=1280, max_height=720, mod=8):
    h, w, c = img.shape
    
    # Determine crop size 
    crop_w = min(w, max_width)
    crop_h = min(h, max_height)
    
    # Set dimensions to prevent tensor size errors 
    crop_w = crop_w - (crop_w % mod)
    crop_h = crop_h - (crop_h % mod)
    
    # Calculate starting coordinates
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    return img[start_y:start_y+crop_h, start_x:start_x+crop_w]

def main():
    # Configuration 
    input_dir = '.' 
    output_dir = './demo_results'
    weights_path = '../pretrained_models/model_deraining.pth' 

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading MPRNet Deraining Model on GPU...")
    model = MPRNet()
    
    checkpoint = torch.load(weights_path, map_location='cuda')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()

    # Get images in the current folder 
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not all_files:
        print("No images found in this folder. Please ensure the photos are in the same folder as this script.")
        return

    print(f"Starting processing of all {len(all_files)} images...")

    with torch.no_grad():
        for i, file_name in enumerate(all_files):
            img_path = os.path.join(input_dir, file_name)
            
            # Load the rainy input image
            img_bgr = cv2.imread(img_path)
            
            # Apply center crop instead of resizing/compressing
            img_bgr = center_crop(img_bgr)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to PyTorch Tensor
            input_tensor = TF.to_tensor(img_rgb).unsqueeze(0).cuda()

            # Pass through MPRNet
            print(f"Processing [{i+1}/{len(all_files)}]: {file_name}")
            restored_tensor = model(input_tensor)[0]
            
            # Convert output back to Image
            restored_img = restored_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            restored_img = np.clip(restored_img, 0, 1) * 255
            restored_img = restored_img.astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

            # Stitch images side-by-side 
            comparison = np.hstack((img_bgr, restored_bgr))
            
            # Add text labels 
            cv2.putText(comparison, 'Original (Cropped)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(comparison, 'MPRNet (Restored)', (img_bgr.shape[1] + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Save the final image
            save_path = os.path.join(output_dir, f"comparison_{file_name}")
            cv2.imwrite(save_path, comparison)

    print(f"\nSuccess! All side-by-side images saved to: {output_dir}")

if __name__ == '__main__':
    main()