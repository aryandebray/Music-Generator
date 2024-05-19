from audiocraft.models import MusicGen
import streamlit as st
import os
import torch
import torchaudio
import numpy as np
import base64


def load_model():
    model=MusicGen.get_pretrained("facebook/musicgen-small")
    return model

def generate_music_tensors(desc,dur:int):
    print("Description: ",desc)
    print("Duration: ",dur)
    model=load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k= 250,
        duration=dur
    )
    output=model.generate(
        descriptions= [desc],
        progress=True,
        return_tokens=True
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    sample_rate=32000
    save_path="AudioOutput/"

    assert samples.dim() ==2 or samples.dim()==3

    samples = samples.detach().cpu()

    if samples.dim()==2:
        samples=samples[None,...]

    for idx,audio in enumerate(samples):
        audio_path=os.path.join(save_path,f"audio_{idx}.wav")
        torchaudio.save(audio_path,audio,sample_rate)

def get_binary_file_down_html(bin_file,file_label='File'):
    with open(bin_file, 'rb') as f:
        data=f.read()
    bin_str=base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def main():
    st.title("Aryan's Music Generator")
    text=st.text_area("Enter your description")
    time_slider=st.slider("Select time duration (in seconds)",2,20,5)
    if text and time_slider:
        st.json(
            {
                "Your Description":text,
                "Time Duration": time_slider
            }
        )
        st.subheader("Generated Music")

        music_tensors=generate_music_tensors(text,time_slider)
        print("Music Tensors: ",music_tensors)
        save_music_file=save_audio(music_tensors)
        audio_filepath='AudioOutput/audio_0.wav'
        audio_file=open(audio_filepath,'rb')
        audio_bytes=audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_down_html(audio_filepath,'Audio'),unsafe_allow_html=True)


if __name__ == "__main__":
    main()