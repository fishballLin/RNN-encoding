# RNN encoding
Generate vector representation of sentences

# Requirement
    *tensorflow 1.2.0+
    *tqdm

# Usage
    1. preprocess
        * put pre_subtitle_no_TC/ into RNN-encoding/data/
        * run bash preprocess.sh

    2. train
        * python3 main.py train

    3. test
        * put AIFirst_test_problem.txt into RNN-encoding/data/
        * python3 main.py test --load_dir <model_path> --result_output <output_file_path>
