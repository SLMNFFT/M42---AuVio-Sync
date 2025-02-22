import streamlit as st
import time
import librosa
import numpy as np
import cv2
import os
import tempfile
import subprocess
import tensorflow as tf  # For utilizing GPU resources (TensorFlow example)

st.set_page_config(layout="wide")

# Ensure TensorFlow uses the GPU if available
if tf.config.list_physical_devices('GPU'):
    st.write("🚀 GPU is available!")
else:
    st.write("⚠️ No GPU detected. Running on CPU.")

def convert_audio_to_wav(input_path, output_path):
    """Convert audio to WAV format using ffmpeg"""
    command = f"ffmpeg -i \"{input_path}\" -ar 44100 -ac 2 -b:a 192k \"{output_path}\" -y"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def analyze_audio(audio_path):
    """Analyze the audio file and extract features like RMS and onset envelope"""
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(rms, sr=sr)
    rms_normalized = (rms - rms.min()) / (rms.max() - rms.min())
    return times, rms_normalized, onset_env, sr

def process_video(video_file, rms_normalized, onset_env, sr, audio_duration, zoom_factor, brightness, contrast, blur_intensity, rotation_angle, max_loop_count, target_fps, effect_sensitivity, apply_cartoon_effect):
    """Process the video file and apply various visual effects based on the audio features"""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    progress_bar = st.progress(0)
    
    frame_count = 0
    required_frames = int(audio_duration * target_fps)
    
    processed_frames = []
    loop_count = 0

    while cap.isOpened() and frame_count < required_frames and loop_count < max_loop_count:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            loop_count += 1
            continue

        effect_intensity = onset_env[frame_count % len(onset_env)] * effect_sensitivity

        # Zoom effect
        if effect_intensity > 0.5:
            zoom_level = zoom_factor + (effect_intensity * 0.3)
            h, w, _ = frame.shape
            new_h, new_w = int(h / zoom_level), int(w / zoom_level)
            y1, x1 = (h - new_h) // 2, (w - new_w) // 2
            frame = frame[y1:y1 + new_h, x1:x1 + new_w]
            frame = cv2.resize(frame, (w, h))

        # Rotation Effect
        if rotation_angle > 0:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation_angle * effect_intensity, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h))

        # Blur Effect
        if blur_intensity > 0:
            blur_value = int(effect_intensity * blur_intensity)
            if blur_value % 2 == 0:
                blur_value += 1  # Ensure odd kernel size
            frame = cv2.GaussianBlur(frame, (blur_value, blur_value), 0)

        # Apply Cartoon Effect if enabled
        if apply_cartoon_effect:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.medianBlur(frame_gray, 7)
            edges = cv2.adaptiveThreshold(frame_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            frame_color = cv2.bilateralFilter(frame, 9, 250, 250)
            frame = cv2.bitwise_and(frame_color, frame_color, mask=edges)

        # Brightness & Contrast Adjustment
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        stframe.image(frame, channels="RGB")
        processed_frames.append(frame)
        
        progress_bar.progress(int((frame_count / required_frames) * 100))  # Update progress bar
        
        time.sleep(1 / target_fps)  # Adjust playback speed
        frame_count += 1
    
    cap.release()
    os.unlink(tfile.name)
    return processed_frames, target_fps

def mix_videos(video_files, target_fps, blend_ratios):
    """Mix multiple videos by blending frames based on provided ratios"""
    processed_frames_list = []
    max_frames = 0
    target_width, target_height = 1920, 1080  # Set a fixed resolution for all videos
    
    progress_bar = st.progress(0)  # Progress bar for video mixing
    
    for video_file in video_files:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = max(max_frames, frame_count)
        processed_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize the frame to the target dimensions (1920x1080)
            frame = cv2.resize(frame, (target_width, target_height))
            processed_frames.append(frame)
        
        processed_frames_list.append(processed_frames)
        cap.release()
        os.unlink(tfile.name)

    # Blend the videos frame by frame based on the provided ratios
    mixed_frames = []
    for i in range(max_frames):
        blended_frame = np.zeros_like(processed_frames_list[0][0], dtype=np.float32)
        
        for j, frames in enumerate(processed_frames_list):
            blend_ratio = blend_ratios[j]
            blended_frame += np.array(frames[i % len(frames)], dtype=np.float32) * blend_ratio
        
        blended_frame = np.clip(blended_frame, 0, 255).astype(np.uint8)
        mixed_frames.append(blended_frame)
        
        progress_bar.progress(int((i / max_frames) * 100))  # Update progress bar for mixing
    
    return mixed_frames

def generate_unique_filename(base_name):
    """Generate a unique filename if the base_name already exists"""
    counter = 1
    while os.path.exists(f"{base_name}_{counter}.mp4"):
        counter += 1
    return f"{base_name}_{counter}.mp4"

def render_final_video(processed_frames, fps, audio_path, output_path):
    """Render the final video with the processed frames and overlay the audio"""
    if not processed_frames:
        st.error("No processed frames available. Ensure the video has frames and try again.")
        return
    
    temp_video = "temp_video.mp4"
    height, width = processed_frames[0].shape[:2]
    
    hd_width, hd_height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (hd_width, hd_height))

    render_progress = st.progress(0)  # Progress bar for rendering

    for i, frame in enumerate(processed_frames):
        frame_resized = cv2.resize(frame, (hd_width, hd_height))
        out.write(cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR))
        render_progress.progress(int((i / len(processed_frames)) * 100))  # Update progress

    out.release()
    
    unique_output_path = generate_unique_filename(output_path)
    command = f"ffmpeg -y -i {temp_video} -i {audio_path} -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k -shortest {unique_output_path}"
    subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.remove(temp_video)
    return unique_output_path

def main():
    st.title("🎵 Audi-VI-Sync V0.3 🎥")
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("⚙️ Setup")
        audio_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav"])
        video_files = st.file_uploader("Upload Videos", type=["mp4", "avi", "mov"], accept_multiple_files=True)
        
        zoom_factor = st.slider("Zoom Effect Intensity", 1.0, 2.0, 1.1, 0.1)
        brightness = st.slider("Brightness Adjustment", -100, 100, 50, 5)
        contrast = st.slider("Contrast Adjustment", 0.5, 2.0, 1.5, 0.1)
        blur_intensity = st.slider("Blur Intensity", 0, 25, 5, 1)
        rotation_angle = st.slider("Rotation Angle", 0, 360, 0, 5)
        max_loop_count = st.slider("Max Video Loop Count", 1, 5, 3, 1)
        target_fps = st.slider("Target Video FPS", 10, 60, 30, 5)
        effect_sensitivity = st.slider("Effect Sensitivity", 0.5, 2.0, 1.0, 0.1)
        apply_cartoon_effect = st.checkbox("Enable Cartoon Effect 🎨")

    with col2:
        st.header("📌 Results")
        processed_frames = []
        fps = 0

        if audio_file and video_files:
            audio_path = "temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())
            convert_audio_to_wav(audio_path, audio_path)
            times, rms_normalized, onset_env, sr = analyze_audio(audio_path)
            audio_duration = librosa.get_duration(path=audio_path)
            
            if len(video_files) > 1:
                blend_ratios = [st.slider(f"Blend Ratio for Video {i+1}", 0.0, 1.0, 1.0, 0.05) for i in range(len(video_files))]
                processed_frames = mix_videos(video_files, target_fps, blend_ratios)
                fps = target_fps
            else:
                processed_frames, fps = process_video(
                    video_files[0], rms_normalized, onset_env, sr, audio_duration,
                    zoom_factor, brightness, contrast, blur_intensity, rotation_angle, max_loop_count, 
                    target_fps, effect_sensitivity, apply_cartoon_effect
                )

        if audio_file and video_files and processed_frames:
            if st.button("🎬 Render Final Video"):
                output_path = "final_output.mp4"
                final_video_path = render_final_video(processed_frames, fps, audio_path, output_path)
                st.video(final_video_path)

if __name__ == "__main__":
    main()

