#!/usr/bin/env bash
sh tools/dist_test.sh \
	--config configs/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py \
	--checkpoint pretrained_model/basicvsr_plusplus_600k_reds4.pth \
	--save-path work_dirs/example_exp/multi_basic++reds \
	--out work_dirs/example_exp/multi_basic++reds.pkl

	
