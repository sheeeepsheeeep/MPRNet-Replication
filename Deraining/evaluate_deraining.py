import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import csv

# Imports the MPRNet architecture 
try:
    from MPRNet import MPRNet
except ImportError:
    print("Error: Could not import MPRNet. Make sure evaluate_deraining.py is inside the 'Deraining' folder.")
    exit()

def load_img(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def main():
    #  Configuration for Deraining
    weights_path = './pretrained_models/model_deraining.pth' 
    
    # Path structure 
    base_dataset_dir = './Datasets/test/' 
    datasets = ['Rain100H', 'Rain100L', 'Test100', 'Test1200', 'Test2800']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Deraining Evaluation on: {device.type.upper()}")

    print("Loading MPRNet architecture and weights...")
    model = MPRNet()
    
    checkpoint = torch.load(weights_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()

    final_results = {}

    with torch.no_grad():
        for dataset_name in datasets:
            print(f"\n" + "="*50)
            print(f"Evaluating Dataset: {dataset_name}")
            print("="*50)
            
            input_dir = os.path.join(base_dataset_dir, dataset_name, 'input')
            target_dir = os.path.join(base_dataset_dir, dataset_name, 'target')
            
            if not os.path.exists(input_dir) or not os.path.exists(target_dir):
                print(f"Skipping {dataset_name}: Directories '{input_dir}' or '{target_dir}' not found.")
                continue

            input_paths = sorted(glob(os.path.join(input_dir, '*.*')))
            
            if len(input_paths) == 0:
                print(f"No images found in {input_dir}")
                continue

            dataset_psnr = []
            dataset_ssim = []

            for inp_path in tqdm(input_paths, desc=f"Processing {dataset_name}"):
                filename = os.path.basename(inp_path)
                tgt_path = os.path.join(target_dir, filename)
                
                if not os.path.exists(tgt_path):
                    continue 
                
                img_input = load_img(inp_path)
                img_target = load_img(tgt_path)
                
                if img_input is None or img_target is None:
                    continue

                input_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(device)

                h, w = input_tensor.shape[2], input_tensor.shape[3]
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                if pad_h != 0 or pad_w != 0:
                    input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), 'reflect')

                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        restored_tensor = model(input_tensor)[0]
                else:
                    restored_tensor = model(input_tensor)[0]

                restored_tensor = restored_tensor[:, :, :h, :w]
                
                del input_tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                restored_img = torch.clamp(restored_tensor, 0, 1).cpu().detach().squeeze(0).permute(1, 2, 0).numpy()

                current_psnr = calculate_psnr(img_target, restored_img, data_range=1.0)
                current_ssim = calculate_ssim(img_target, restored_img, data_range=1.0, channel_axis=2)

                dataset_psnr.append(current_psnr)
                dataset_ssim.append(current_ssim)

            avg_psnr = np.mean(dataset_psnr) if dataset_psnr else 0
            avg_ssim = np.mean(dataset_ssim) if dataset_ssim else 0
            
            final_results[dataset_name] = {'PSNR': avg_psnr, 'SSIM': avg_ssim}
            
            print(f"\n{dataset_name} Results -> Average PSNR: {avg_psnr:.4f} dB | Average SSIM: {avg_ssim:.4f}")

    # Export Results to CSV
    csv_filename = "MPRNet_Deraining_Results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Average PSNR (dB)", "Average SSIM"])
        for name, metrics in final_results.items():
            writer.writerow([name, f"{metrics['PSNR']:.4f}", f"{metrics['SSIM']:.4f}"])
            
    print("\n" + "="*50)
    print("              FINAL EVALUATION SUMMARY")
    print("="*50)
    for name, metrics in final_results.items():
        print(f"{name:<15} | PSNR: {metrics['PSNR']:>7.4f} dB | SSIM: {metrics['SSIM']:>6.4f}")
    print("="*50)
    print(f"Success! Results have been saved to {csv_filename} in the current folder.")

if __name__ == '__main__':
    main()