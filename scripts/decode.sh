# Beam Search decoding
./attentional --beam 5 -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -T ../sample-data/test.de-en.unk.cap -i ./work/exp/params.de-en.AM.l_2_h_512_a_256_lstm_bidirectional --lstm --bidirectional -l 2 -h 512 -a 256 &>./work/res/translation.beam.de-en.AM.l_2_h_512_a_256_lstm_bidirectional & #with beam_width=5

# Greedy Viterbi decoding
./attentional -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -T ../sample-data/test.de-en.unk.cap -i ./work/exp/params.de-en.AM.l_2_h_512_a_256_lstm_bidirectional --lstm --bidirectional -l 2 -h 512 -a 256 &>./work/res/translation.greedy.de-en.AM.l_2_h_512_a_256_lstm_bidirectional &

# Sampling decoding
 


