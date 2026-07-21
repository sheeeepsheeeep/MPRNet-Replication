import os
import random
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as TF

# Import model
from MPRNet import MPRNet

def main():
    # Configuration 
    # Pointing to the SIDD folder structure we fixed earlier
    input_dir = './Datasets/SIDD/Val/input'  
    weights_path = './pretrained_models/model_denoising.pth'
    output_dir = './demo_results'
    num_images = 10

    os.makedirs(output_dir, exist_ok=True)

    # Load Model 
    print("Loading MPRNet Denoising Model on GPU...")
    model = MPRNet()
    checkpoint = torch.load(weights_path, map_location='cuda')
    
    # Unwrap the weights 
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.cuda()
    model.eval()

    # Get 10 Random Images 
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    demo_files = random.sample(all_files, min(num_images, len(all_files)))

    print(f"Starting processing of {len(demo_files)} random noisy images...")

    with torch.no_grad():
        for i, file_name in enumerate(demo_files):
            img_path = os.path.join(input_dir, file_name)
            
            # Load the noisy input image
            img_bgr = cv2.imread(img_path)
            
            # Downscale to prevent GPU freezing 
            img_bgr = cv2.resize(img_bgr, (640, 360))
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to PyTorch Tensor
            input_tensor = TF.to_tensor(img_rgb).unsqueeze(0).cuda()

            # Pass through MPRNet
            print(f"Processing [{i+1}/{len(demo_files)}]: {file_name}")
            restored_tensor = model(input_tensor)[0]
            
            # Convert output back to Image
            restored_img = restored_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            restored_img = np.clip(restored_img, 0, 1) * 255
            restored_img = restored_img.astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

            # Calculate the mathematical difference between noisy and denoised images
            noise_map = cv2.absdiff(img_bgr, restored_bgr)
            
            # Amplify the static noise 
            amplified_noise = cv2.convertScaleAbs(noise_map, alpha=10.0)
            
            # Save the isolated noise map image
            noise_save_path = os.path.join(output_dir, f"noise_removed_{file_name}")
            cv2.imwrite(noise_save_path, amplified_noise)

            # Stitch images side-by-side 
            if img_bgr.shape == restored_bgr.shape:
                comparison = np.hstack((img_bgr, restored_bgr))
                
                # Add text labels
                cv2.putText(comparison, 'Original (Noisy)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(comparison, 'MPRNet (Denoised)', (img_bgr.shape[1] + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Save the final image
                save_path = os.path.join(output_dir, f"comparison_{file_name}")
                cv2.imwrite(save_path, comparison)
            else:
                print(f"Skipping {file_name} due to dimension mismatch.")

    print(f"\nSuccess! All side-by-side images saved to: {output_dir}")

if __name__ == '__main__':
    main()