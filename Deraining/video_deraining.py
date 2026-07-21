import cv2
import torch
import numpy as np
from torchvision.transforms import functional as TF
from MPRNet import MPRNet

def process_video(input_video_path, output_video_path, weights_path):
    # Load Deraining Model
    print("Loading MPRNet Deraining Model...")
    model = MPRNet()
    checkpoint = torch.load(weights_path, map_location='cuda')
    
    # Unwrap the weights safely
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.cuda()
    model.eval()

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Resize to 640x360 
    width, height = 640, 360  
    
    # Update video writer for side-by-side comparison 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {frame_count}")

    current_frame = 0
    with torch.no_grad():
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame += 1
            # Print an update every 10 frames to keep the terminal clean
            if current_frame % 10 == 0 or current_frame == 1:
                print(f"Processing frame {current_frame}/{frame_count}")

            # Resize the frame to fit in VRAM
            img_bgr = cv2.resize(frame, (width, height))

            # Magnify rain streak before passing to model
            # Convert to LAB color space to enhance lightness
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)

            # Merge the enhanced L channel back with A and B channels
            merged_lab = cv2.merge((cl, a_channel, b_channel))
            enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
            

            # Pass the enhanced image to model
            img_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and pass to model
            input_tensor = TF.to_tensor(img_rgb).unsqueeze(0).cuda()
            restored_tensor = model(input_tensor)[0]

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and pass to model
            input_tensor = TF.to_tensor(img_rgb).unsqueeze(0).cuda()
            restored_tensor = model(input_tensor)[0]
            
            # Convert back to image
            restored_img = restored_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            restored_img = np.clip(restored_img, 0, 1) * 255
            restored_bgr = restored_img.astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_bgr, cv2.COLOR_RGB2BGR)

            # Stitch input and output side by side for comparison
            comparison = np.hstack((img_bgr, restored_bgr))
            
            # Add text labels
            cv2.putText(comparison, 'Original (Rainy)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(comparison, 'MPRNet (Derained)', (width + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Write frame to new video
            out.write(comparison)

    cap.release()
    out.release()
    print(f"\nSuccess! Side-by-side video saved to: {output_video_path}")

if __name__ == '__main__':
    # Configurations
    input_vid = 'demo_deraining_input.avi'  
    output_vid = 'demo_deraining_result.avi'
    weights = './pretrained_models/model_deraining.pth'
    
    process_video(input_vid, output_vid, weights)