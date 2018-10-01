python3 -u search_classification_plus.py -m gru -bi
mv results bi_random_results
mv tmp_pkl bi_random_tmp_pkl
python3 -u search_classification_plus.py -m gru -em charngram -bi -f
mv results bi_charngram_results
mv tmp_pkl bi_charngram_tmp_pkl
python3 -u search_classification_plus.py -m gru -em fasttextEn -bi -f
mv results bi_fasttextEn_results
mv tmp_pkl bi_fasttextEn_tmp_pkl
python3 -u search_classification_plus.py -m gru -em fasttextSimple -bi -f
mv results bi_fasttextSimple_results
mv tmp_pkl bi_fasttextSimple_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove42 -bi -f
mv results bi_glove42_results
mv tmp_pkl bi_glove42_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove84 -bi -f
mv results bi_glove84_results
mv tmp_pkl bi_glove84_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter25 -bi -f
mv results bi_gloveTwitter25_results
mv tmp_pkl bi_gloveTwitter25_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter50 -bi -f
mv results bi_gloveTwitter50_results
mv tmp_pkl bi_gloveTwitter50_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter100 -bi -f
mv results bi_gloveTwitter100_results
mv tmp_pkl bi_gloveTwitter100_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter200 -bi -f
mv results bi_gloveTwitter200_results
mv tmp_pkl bi_gloveTwitter200_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_80 -bi -f
mv results bi_glove6b_80_results
mv tmp_pkl bi_glove6b_80_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_100 -bi -f
mv results bi_glove6b_100_results
mv tmp_pkl bi_glove6b_100_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_200 -bi -f
mv results bi_glove6b_200_results
mv tmp_pkl bi_glove6b_200_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_300 -bi -f
mv results bi_glove6b_300_results
mv tmp_pkl bi_glove6b_300_tmp_pkl