# Experiment configs

For our paper, we run three different sets of experiments based on the underlying data used: *synthetic*, *Alpaca Eval*, and *Chatbot Arena*.

To reproduce our results use the following commands:

- Synthetic results:
  - Run the following commands six times (we run each six times to average results over for seeds)
```
icai-exp -cd ./exp/configs/001_synthetic_orthogonal
icai-exp -cd ./exp/configs/002_synthetic_aligned
icai-exp -cd ./exp/configs/003_synthetic_unaligned
```

- AlpacaEval results:
  - Run the following three commands, each six times:
```
icai-exp -cd ./exp/configs/011_alpacaeval_aligned_gpt4o
icai-exp -cd ./exp/configs/013_alpacaeval_unaligned_gpt4o
icai-exp -cd ./exp/configs/013_alpacaeval_unaligned_gpt4o_crossmodel
```

- Chatbot Arena results
  - Run the following two commands, each six times
```
icai-exp -cd ./exp/configs/021_arena_cross_user
icai-exp -cd ./exp/configs/022_arena_cross_user_opposite
```

For each separate run, a new directory in `exp/outputs` will be created. Combine the runs of the same commands into a single folder to enable the plotting. Then adapt the plotting notebooks in `notebooks` to your new experimental folders. Finally, use the `notebooks/10_plots_full_paper.ipynb` to plot all relevant plots in the paper.
