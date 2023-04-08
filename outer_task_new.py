import multiprocessing
import os


def process_metadata_file(meta_file):
    input_files_prefix = "/data3/nordic-pile/txt-files-all"
    vocab_file = "/data3/vocab.txt"
    output_dir = "/data3/training_data/train_512k_uncased_msl512"
    max_seq_length = 512
    max_predictions_per_seq = 80
    spacy_model = "sv_core_news_lg"
    overlap_size = 0

    with open(meta_file) as f:
        file_paths = f.readlines()

    for file_path in file_paths:
        # run the command for each file_path
        cmd = f"python create_csv_mlm_only.py " \
              f"--metadata_files {meta_file} " \
              f"--input_files_prefix {input_files_prefix} " \
              f"--vocab_file {vocab_file} " \
              f"--output_dir {output_dir} " \
              f"--max_seq_length {max_seq_length} " \
              f"--max_predictions_per_seq {max_predictions_per_seq} " \
              f"--spacy_model {spacy_model} " \
              f"--overlap_size {overlap_size} {file_path.strip()}"
        os.system(cmd)


if __name__ == '__main__':
    meta_files_dir = "/data3/nordic-pile/train_meta"
    meta_files = [os.path.join(meta_files_dir, f) for f in os.listdir(meta_files_dir)]

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    pool.map(process_metadata_file, meta_files)

    pool.close()
    pool.join()
