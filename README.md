Audi-VI-Sync V0.3
Audi-VI-Sync V0.3 is a Streamlit-based tool that synchronizes video effects with audio features. It processes uploaded audio and video files, extracting key sound properties like RMS and onset strength using Librosa. These features then dynamically control video effects such as zoom, rotation, blur, brightness, contrast, and even a cartoon filter.
The tool supports GPU acceleration via TensorFlow for optimized performance. It converts uploaded audio to WAV format for analysis and allows users to apply customizable visual effects. The video processing function loops footage if necessary and adjusts effects in real-time based on audio characteristics.
Additionally, Audi-VI-Sync can blend multiple videos by mixing frames at user-defined ratios, ensuring smooth transitions. The final processed video is rendered in high resolution (1920x1080) with synchronized audio and can be exported in MP4 format using FFmpeg.
With an intuitive Streamlit UI, users can fine-tune parameters like zoom intensity, blur, contrast, rotation, effect sensitivity, and looping limits. The progress bar provides real-time feedback on processing, mixing, and rendering stages.
This tool is ideal for content creators looking to enhance videos with dynamic, audio-driven effects without requiring advanced editing software. 🚀

License Agreement

Audi-VI-Sync V0.3 is provided under a Non-Commercial Use License.

    You may use, modify, and distribute this tool for personal, educational, or research purposes without any charge.
    Commercial use of this software, including but not limited to monetized videos, advertisements, corporate projects, or paid services, requires explicit permission and a royalty agreement with the developer.
    Redistribution or modification for commercial gain without authorization is strictly prohibited.

For commercial licensing inquiries, please contact the developer.


Installation Guide for Audi-VI-Sync V0.3

1️⃣ System Requirements
    OS: Windows, macOS, or Linux
    Python: 3.8 or higher
    GPU Acceleration (Optional): NVIDIA CUDA for TensorFlow

2️⃣ Install Dependencies

First, create a virtual environment (recommended):

python -m venv audi_vi_sync_env
source audi_vi_sync_env/bin/activate  # macOS/Linux
audi_vi_sync_env\Scripts\activate    # Windows

Then, install required libraries:

pip install streamlit opencv-python numpy librosa moviepy ffmpeg-python tensorflow tqdm

For GPU acceleration (optional):

pip install tensorflow-gpu

3️⃣ Install FFmpeg
FFmpeg is required for video processing.
    Windows: Download FFmpeg from https://ffmpeg.org/download.html
        Add ffmpeg/bin to the system PATH.
    Linux/macOS: Install via package manager:
    sudo apt install ffmpeg  # Ubuntu/Debian
    brew install ffmpeg      # macOS (Homebrew)

4️⃣ Run the Application

Start the Streamlit app:
streamlit run app.py
This will launch Audi-VI-Sync in your browser, where you can upload audio/video files and apply synchronized effects! 🎬
Let me know if you need additional setup help. 🚀
