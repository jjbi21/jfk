Subword and WordRNN Implementation

Note: Trained Models and Vector Embeddings are provided for generating Subword RNN comments for all categories
.pkl files contain lists of categorized comments

Sample Training Script
python train.py comments.pkl --chunk_len 80 --batch_size 80 --n_epochs 8000 --n_layers 2 --cuda --p

Sample Generation Script
python generate.py comments.pt --punc --cuda

By default, priming string of "I" is used between training epochs to generate sample text.

Flags --p and --punc should be used for training Subword RNN, but dropped for WordRNN


NOTE:
Most of the files are artifacts from pretraining the subword model