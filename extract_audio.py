from moviepy.editor import *
import os
import argparse
import cv2
import time
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract(dataset):
    dataset = dataset.upper()
    input_directory_path = f'data/{dataset}/Raw'
    output_directory_path = f'data/{dataset}/wav'
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    for folder in tqdm(os.listdir(input_directory_path)):
        
        input_folder_path = os.path.join(input_directory_path, folder)
        output_folder_path = os.path.join(output_directory_path, folder)
        if not os.path.exists(output_folder_path):  
            os.makedirs(output_folder_path)
        
        for file_name in os.listdir(input_folder_path):
            if file_name.split(".")[-1] != "mp4" or file_name.split(".")[1] != "mp4":
                continue
            input_file_path = os.path.join(input_folder_path, file_name)
            output_file_path = os.path.join(output_folder_path, file_name)
            if "-edited.mp4" in output_file_path:
                output_file_path = output_file_path.replace("-edited.mp4", ".mp4")
            output_file_path = output_file_path.replace(".mp4", ".wav")
            # Skip if the video file is already edited
            if os.path.exists(input_file_path.replace(".mp4", "-edited.mp4")):
                continue
            # Skip if the audio file already exists
            if os.path.exists(output_file_path):
                continue
            try:
                # Load the video file
                video = VideoFileClip(input_file_path)

                # Extract the audio from the video
                audio = video.audio

                # Set the desired sampling rate
                desired_sampling_rate = 16000  # Replace this value with your desired sampling rate

                # Resample the audio to the desired sampling rate
                resampled_audio = audio.set_fps(desired_sampling_rate)
                
                # Save the extracted and resampled audio to a WAV file
                resampled_audio.write_audiofile(output_file_path, codec='pcm_s16le', verbose=False, logger=None)
            except Exception as e:
                # one edited video lost audio, so we extract audio from the original video instead
                # there are also 6 videos that are corrupted, so we skip them
                print(input_file_path, e)
                if "-edited.mp4" in input_file_path:
                    input_file_path = input_file_path.replace("-edited.mp4", ".mp4")
                    video = VideoFileClip(input_file_path)
                    audio = video.audio
                    desired_sampling_rate = 16000
                    resampled_audio = audio.set_fps(desired_sampling_rate)
                    resampled_audio.write_audiofile(output_file_path, codec='pcm_s16le', verbose=False, logger=None)


def preprocess_video_file(filename):
    start = time.time()
    cap = cv2.VideoCapture(filename)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_counter = 0
    for f in range(n_frames):
        ret, frame = cap.read()
        frame_counter += 1
        if ret:
            continue
        elif frame_counter > n_frames:
            return None
        else:
            duration = (frame_counter - 1) / fps
            print(f"Fixing bad video file: {filename}")
            # print(n_frames, frame_counter, duration, '\n')            
            return duration
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sims', help='dataset name')
    args = parser.parse_args()

    # fix the video duration of MOSEI dataset: lots of videos have bad frames at the end
    if args.dataset == 'mosei':
        print("Fixing MOSEI video files!")
        invalid_files = ['3aIQUQgawaI/12', '94ULum9MYX0/2', 'mRnEJOLkhp8/24', 'aE-X_QdDaqQ/3', '94ULum9MYX0/11', 'mRnEJOLkhp8/26']
        directory_path = 'data/MOSEI/Raw'
        for folder in os.listdir(directory_path):
            folder_path = os.path.join(directory_path, folder)
            for file_name in os.listdir(folder_path):
                fpath = os.path.join(folder_path, file_name)
                if "-edited.mp4" in fpath:
                    continue
                if os.path.exists(fpath.replace(".mp4", "-edited.mp4")):
                    continue
                if os.path.join(folder, file_name.split(".")[0]) in invalid_files:
                    continue
                duration = preprocess_video_file(fpath)

                if duration:
                    with VideoFileClip(fpath) as video:
                        new = video.subclip(0, duration)
                        new.write_videofile(fpath.replace(".mp4", "-edited.mp4"), verbose=False, logger=None)

    extract(args.dataset)