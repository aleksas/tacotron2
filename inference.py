import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import os

from hparams import create_hparams
from train import load_model
from text import text_to_sequence
import matplotlib.pylab as plt

def plot_data(data, filename='myfilename', figsize=(32, 8)):
    fig, axes = plt.subplots(len(data), 1, figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='upper', 
                       interpolation='none')
    plt.savefig(filename + '.png', dpi=200)


hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "outdir/checkpoint_90000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()


infere_single_audio = False

if infere_single_audio:
    waveglow_path = 'waveglow_256channels_universal_v5.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()

    speaker_id = 0

    text = "Waveglow is really awesome! and use the patch tool to revert the changes."
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
        
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, speaker_id)

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        
    import soundfile as sf

    sf.write('output_audio3.wav', audio[0].data.cpu().numpy(), hparams.sampling_rate)
else:    
    filelists = [
        "./filelists/ljs_audio_text_test_filelist.txt",
        "./filelists/ljs_audio_text_train_filelist.txt",
        "./filelists/ljs_audio_text_val_filelist.txt",
    ]
    
    all_lines = []
    
    for file_path in filelists:
        with open(file_path, 'r') as file:
            all_lines += file.readlines()
            
            
    count_lines = len(all_lines)
    
    for i, line in enumerate(all_lines):      
        if not line.strip():
            continue
        
        path, text = line.strip().split('|')
        
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        tensor = mel_outputs_postnet[0].detach().cpu().numpy()
        mel  = np.load(path + ".npy")
        plot_data((tensor, mel), path)
        if True or not os.path.isfile(path + ".infere.npy"):
            np.save(path + ".infere.npy", tensor)
        print (f'{i}/{count_lines}  {path}')