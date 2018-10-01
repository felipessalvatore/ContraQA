python3 -u search_classification_plus.py -m gru -em charngram
mv results charngram_results
mv tmp_pkl charngram_tmp_pkl
python3 -u search_classification_plus.py -m gru -em fasttextEn
mv results fasttextEn_results
mv tmp_pkl fasttextEn_tmp_pkl
python3 -u search_classification_plus.py -m gru -em fasttextSimple
mv results fasttextSimple_results
mv tmp_pkl fasttextSimple_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove42
mv results glove42_results
mv tmp_pkl glove42_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove84
mv results glove84_results
mv tmp_pkl glove84_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter25
mv results gloveTwitter25_results
mv tmp_pkl gloveTwitter25_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter50
mv results gloveTwitter50_results
mv tmp_pkl gloveTwitter50_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter100
mv results gloveTwitter100_results
mv tmp_pkl gloveTwitter100_tmp_pkl
python3 -u search_classification_plus.py -m gru -em gloveTwitter200
mv results gloveTwitter200_results
mv tmp_pkl gloveTwitter200_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_80
mv results glove6b_80_results
mv tmp_pkl glove6b_80_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_100
mv results glove6b_100_results
mv tmp_pkl glove6b_100_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_200
mv results glove6b_200_results
mv tmp_pkl glove6b_200_tmp_pkl
python3 -u search_classification_plus.py -m gru -em glove6b_300
mv results glove6b_300_results
mv tmp_pkl glove6b_300_tmp_pkl