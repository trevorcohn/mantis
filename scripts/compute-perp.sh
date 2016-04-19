# get the scores from n-best list
# assume that the n-best list should follow the format: 
# line in source language ||| line in target language
# and all sentence pairs are wrapped up with <s>, </s>, <unk> markers
./attentional -t ../sample-data/train.de-en.unk.cap -d ../sample-data/dev.de-en.unk.cap -T ../sample-data/50best.de-en.unk.cap -i ./work/exp/params.de-en.AM.l_2_h_512_a_256_lstm_bidirectional --lstm --bidirectional -l 2 -h 512 -a 256 --rescore &>./work/res/scores.50best.de-en.AM.l_2_h_512_a_256_lstm_bidirectional & 

# note that if using advanced models (with alignment features), please follow the usage of extra params in train.sh

