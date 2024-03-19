import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import threading
import os
import whisper 

class AudioRecorder:
    def __init__(self, filename="output.wav"):
        self.filename = filename
        self.recording = False

    def start_gui(self):
        self.window = tk.Tk()
        self.window.title("Audio Recorder")
        label = tk.Label(self.window, text="Click the button to start recording")
        label.pack()

        self.start_button = tk.Button(self.window, text="Start Recording", command=self.start_recording)
        self.start_button.pack()

        self.stop_button = tk.Button(self.window, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack()

        self.window.mainloop()

    def start_recording(self):
        self.recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        thread = threading.Thread(target=self.record)
        thread.start()

    def stop_recording(self):
        self.recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        messagebox.showinfo("Recording stopped", "The audio recording was saved.")
        self.window.destroy()

    def record(self):
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100
        p = pyaudio.PyAudio()
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
        frames = []

        while self.recording:
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

def get_text_from_audio(filename="output.wav"):
    recorder = AudioRecorder(filename)
    recorder.start_gui()
    file_path = os.path.abspath(filename)
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]


# # Example usage
# if __name__ == "__main__":
#     text_from_audio = get_text_from_audio("current_recording.wav")
#     print(f"Audio transcribed from text: {text_from_audio}")