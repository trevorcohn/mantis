## train vanilla attentional models

# print help (no params)
./attentional

# forward model
./attentional -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -p ./work/exp/params.de-en.AM.l_2_h_512_a_256_lstm_bidirectional --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.de-en.AM.train.l_2_h_512_a_256_lstm_bidirectional & 

# backward model (simply add '--swap' param)
./attentional --swap -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -p /work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.en-de.AM.train.l_2_h_512_a_256_lstm_bidirectional & 

## train attentional model with alignment features (including positional bias, Markov conditioning, and global/local fertilities)

# with positional bias (simple add '--gz-position' param)
./attentional --gz-position -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -p /work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional_giza-position --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.en-de.AM.train.l_2_h_512_a_256_lstm_bidirectional_giza-position &

# with Markov condition (simple add '--gz-condition' param)
./attentional --gz-markov -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -p /work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional_giza-markov --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.en-de.AM.train.l_2_h_512_a_256_lstm_bidirectional_giza-markov &

# with local fertility (simple add '--gz-fertility' param)
./attentional --gz-fertility -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -p /work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional_giza-fertility --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.en-de.AM.train.l_2_h_512_a_256_lstm_bidirectional_giza-fertility &

# with all the above features (simple add '--giza' param)
./attentional --giza -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -p /work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional_giza-all --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.en-de.AM.train.l_2_h_512_a_256_lstm_bidirectional_giza-all &

## train attentional model with global fertility
# First, we need a pre-trained model with (better) or without alignment features using the commands mentioned earlier
# Next, let's train the model with global fertility (simple add '--fertility' param)
./attentional --giza --fertility -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -i /work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional_giza-all --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.en-de.AM.train.l_2_h_512_a_256_lstm_bidirectional_giza-all_continued_fertility &

# train bi-attentional model
# First, we need pre-trained models with both directions (forward and backward)
# e.g. forward model: ./work/exp/params.de-en.AM.l_2_h_512_a_256_lstm_bidirectional
# and backward model: ./work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional
./biattentional -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -i ./work/exp/params.de-en.AM.l_2_h_512_a_256_lstm_bidirectional -i /work/exp/params.en-de.AM.l_2_h_512_a_256_lstm_bidirectional -p /work/exp/params.de-en_en-de.AM.l_2_h_512_a_256_lstm_bidirectional_symmetry --lstm --bidirectional -l 2 -h 512 -a 256 -v &>/work/exp/log.de-en_en-de.AM.train.l_2_h_512_a_256_lstm_bidirectional_symmetry &


