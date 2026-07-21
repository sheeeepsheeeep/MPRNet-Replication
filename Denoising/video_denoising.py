import cv2
import torch
import numpy as np
from torchvision.transforms import functional as TF
from MPRNet import MPRNet

def process_denoise_video(input_video_path, output_video_path, weights_path):
    # Load the Denoising Model 
    print("Loading MPRNet Denoising Model...")
    model = MPRNet()
    checkpoint = torch.load(weights_path, map_location='cuda')
    
    # Safely unwrap weights 
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.cuda()
    model.eval()

    # Open the Grainy Video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    # Get original dimensions 
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #  1:1 Native Center Crop
    crop_w, crop_h = 640, 360
    
    if orig_width < crop_w or orig_height < crop_h:
        print(f"Error: Video dimensions ({orig_width}x{orig_height}) are too small for a {crop_w}x{crop_h} crop.")
        return

    # Preserves raw pixel noise without blending artifacts 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (crop_w * 2, crop_h))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {frame_count}")

    current_frame = 0
    with torch.no_grad():
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame += 1
            if current_frame % 10 == 0 or current_frame == 1:
                print(f"Processing frame {current_frame}/{frame_count}")

            # Calculate center slice coordinates
            start_y = orig_height // 2 - crop_h // 2
            start_x = orig_width // 2 - crop_w // 2
            
            # Extract the raw & uncompressed grainy window
            img_bgr = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert frame to PyTorch tensor 
            input_tensor = TF.to_tensor(img_rgb).unsqueeze(0).cuda()
            restored_tensor = model(input_tensor)[0]
            
            # Convert clean tensor back to image array
            restored_img = restored_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            restored_img = np.clip(restored_img, 0, 1) * 255
            restored_bgr = restored_img.astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_bgr, cv2.COLOR_RGB2BGR)

            # Stitch Side-by-Side 
            comparison = np.hstack((img_bgr, restored_bgr))
            
            # Add text headers 
            cv2.putText(comparison, 'Original (Grainy Noise)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(comparison, 'MPRNet (Denoised)', (crop_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write the unified frame 
            out.write(comparison)

    cap.release()
    out.release()
    print(f"\nSuccess! Side-by-side denoising video saved to: {output_video_path}")

if __name__ == '__main__':
    # Configurations
    input_vid = 'demo_denoising_input.avi'  
    output_vid = 'demo_denoising_result.avi'
    weights = './pretrained_models/model_denoising.pth'
    
    process_denoise_video(input_vid, output_vid, weights)