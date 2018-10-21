# expected time 4.52 days, 7.75 hours per search

python3 -u search_classification_inf.py -m gru -f
mv results random_results
mv tmp_pkl random_tmp_pkl
python3 -u search_classification_inf.py -m gru -em charngram -f
mv results charngram_results
mv tmp_pkl charngram_tmp_pkl
python3 -u search_classification_inf.py -m gru -em fasttextEn -f
mv results fasttextEn_results
mv tmp_pkl fasttextEn_tmp_pkl
python3 -u search_classification_inf.py -m gru -em fasttextSimple -f
mv results fasttextSimple_results
mv tmp_pkl fasttextSimple_tmp_pkl
python3 -u search_classification_inf.py -m gru -em glove42 -f
mv results glove42_results
mv tmp_pkl glove42_tmp_pkl
python3 -u search_classification_inf.py -m gru -em glove84 -f
mv results glove84_results
mv tmp_pkl glove84_tmp_pkl
python3 -u search_classification_inf.py -m gru -em gloveTwitter25 -f
mv results gloveTwitter25_results
mv tmp_pkl gloveTwitter25_tmp_pkl
python3 -u search_classification_inf.py -m gru -em gloveTwitter50 -f
mv results gloveTwitter50_results
mv tmp_pkl gloveTwitter50_tmp_pkl
python3 -u search_classification_inf.py -m gru -em gloveTwitter100 -f
mv results gloveTwitter100_results
mv tmp_pkl gloveTwitter100_tmp_pkl
python3 -u search_classification_inf.py -m gru -em gloveTwitter200 -f
mv results gloveTwitter200_results
mv tmp_pkl gloveTwitter200_tmp_pkl
python3 -u search_classification_inf.py -m gru -em glove6b_80 -f
mv results glove6b_80_results
mv tmp_pkl glove6b_80_tmp_pkl
python3 -u search_classification_inf.py -m gru -em glove6b_100 -f
mv results glove6b_100_results
mv tmp_pkl glove6b_100_tmp_pkl
python3 -u search_classification_inf.py -m gru -em glove6b_200 -f
mv results glove6b_200_results
mv tmp_pkl glove6b_200_tmp_pkl
python3 -u search_classification_inf.py -m gru -em glove6b_300 -f
mv results glove6b_300_results
mv tmp_pkl glove6b_300_tmp_pkl