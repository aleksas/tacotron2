
1. Download [LJ Speech Data]. In this example it's in `data/`

2. Make a list of the file names to use for training/testing

   ```command
   ls data/*.wav > finetune_files.txt
   ```

3. Infere mel spectrograms

   ```command
   python infere.py finetune_files.txt
   ```