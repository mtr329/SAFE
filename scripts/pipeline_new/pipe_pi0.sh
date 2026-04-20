./scripts/pipeline_new/submit_pi0diff_libero.bash 

./scripts/pipeline_new/val_new.sh --gpu 0 --logs-dir log_ckpt_new/pizero --method trans

./scripts/pipeline_new/test_new.sh --logs-dir log_ckpt_new/pizero

python scripts/export_selection_barplots_to_latex.py log_ckpt_new/pizero/pipeline_test_new/selection_barplots/selection_barplot_summary.csv