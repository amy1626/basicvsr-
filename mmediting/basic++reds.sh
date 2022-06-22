python tools/test.py \
	--config configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py \
	--checkpoint pretrained_model/iter_600000.pth \
	--out work_dirs/example_exp/results_basic++shigu.pkl \
	--save-path work_dirs/example_exp/results_basic++shigu/


#--checkpoint pretrained_model/iter_600000.pth 
#--checkpoint pretrained_model/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
#--checkpoint work_dirs/basicvsr_plusplus_c64n7_8x1_600k_reds4/iter_1000.pth

python tools/test.py \
	--config configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py \
	--checkpoint work_dirs/basicvsr_plusplus_c64n7_8x1_600k_reds4/iter_1000.pth \
	--out work_dirs/example_exp/results_basic++4k.pkl \
	--save-path work_dirs/example_exp/results_basic++4k/

