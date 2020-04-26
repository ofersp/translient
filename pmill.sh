rm -f figures/* output/*
papermill stamps.ipynb output/stamps_sims_1.ipynb -p sess_name sims_1
papermill roc_curves.ipynb output/roc_curves_sims_1_2_3.ipynb -p sess_name_a sims_1 -p sess_name_b sims_2 -p sess_name_c sims_3
papermill roc_curves.ipynb output/roc_curves_sims_4_5_6.ipynb -p sess_name_a sims_4 -p sess_name_b sims_5 -p sess_name_c sims_6
