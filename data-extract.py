import os
import lzma
from tqdm import tqdm


def xz_files_in_dir(direcroty):
    files = []
    for f in os.listdir(direcroty):
        if f.endswith('.xz') and os.path.isfile(os.path.join(direcroty, f)):
            files.append(f)
    return files


def file_processing(folder_path, output_file, file_batch, vocab_set):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in tqdm(file_batch, total=len(file_batch)):
            file_path = os.path.join(folder_path, filename)
            with lzma.open(file_path, 'rt', encoding='utf-8') as infile:
                text = infile.read()
                outfile.write(text)
                characters = set (text)
                vocab_set.update(characters)
    return


def main() -> None:

    folder_path = 'data/openwebtext'
    output_file_train = 'data/output_train.txt'
    output_file_val = 'data/output_val.txt'
    vocab_file = 'data/vocab.txt'

    files = xz_files_in_dir(folder_path)
    total_files = len(files)

    # calc the split indices
    split_index = int(total_files * 0.9)
    files_train = files[:split_index]
    files_val = files[split_index:]

    # process the files for training / validation split 
    vocab = set()
    file_processing(folder_path, output_file_train, files_train, vocab)
    file_processing(folder_path, output_file_val, files_val, vocab)


    # Write vocab to vocab.txt
    with open(vocab_file, 'w', encoding='utf-8') as vfile:
        for char in vocab:
            vfile.write(char + '\n')


    return 

if __name__ == '__main__':
    main()