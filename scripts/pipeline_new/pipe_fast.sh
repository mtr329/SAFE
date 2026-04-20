./scripts/pipeline_new/submit_pi0fast_libero.bash  --gpus 0 --logs-dir log_ckpt_new/pizero_fast

./scripts/pipeline_new/val_new.sh --gpu 0 --logs-dir log_ckpt_new/pizero_fast --method trans

./scripts/pipeline_new/test_new.sh --logs-dir log_ckpt_new/pizero_fast

python scripts/export_selection_barplots_to_latex.py log_ckpt_new/pizero_fast/pipeline_test_new/selection_barplots/selection_barplot_summary.csv