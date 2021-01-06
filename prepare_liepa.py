import os
import random
import math
import argparse

def get_data(voice, subfolder, transcript_list):
    with open(os.path.join(voice, transcript_list), 'rt', encoding='utf-8') as fp:
        for idx, text in enumerate(fp.readlines()):
            yield os.path.join(voice, subfolder, "%d.wav" % idx), text

def get_subsets(proportions, voice, subfolder, transcript_list):
    data = list(get_data(voice, subfolder, transcript_list))
    random.shuffle(data)
    start = 0
    for i, p in enumerate(proportions):
        p = p/sum(proportions)
        end = start + int(math.floor(len(data) * p)) if i < len(proportions) - 1 else len(data)

        yield data[start:end]

        start = end

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare liepa dataset fot training.')
    parser.add_argument('-v', '--voice', default='Regina', dest='voice', help="One of available voices [Regina, Aiste, Vladas, Edvardas]")
    parser.add_argument('-t', '--transcript-list', default='db_tr.txt', dest='transcript_list', help="File containing lines of text recorded audio files with line number corresponding to audio file name.$

    args = parser.parse_args()

    dataset = "Liepa"
    rel_dir = "filelists"
    param = rel_dir, dataset, args.voice

    sets_metadata = [
        ("%s/%s_%s_audio_text_train_filelist.txt" % param, 94), 
        ("%s/%s_%s_audio_text_val_filelist.txt" % param, 1),
        ("%s/%s_%s_audio_text_test_filelist.txt" % param, 5)
    ]

    os.makedirs(rel_dir, exist_ok=True)

    proportions = [p for _, p in sets_metadata]

    subsets = get_subsets(proportions=proportions, voice=args.voice, subfolder="data", transcript_list=args.transcript_list)

    for path, subset in zip([p for p,_ in sets_metadata], subsets):
        with open(path, 'wt', encoding='utf-8') as fp:
            for audio_path, text in subset:
                line = "%s|%s" % (audio_path, text)
                fp.write(line)


