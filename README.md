# Master thesis
This repository is created for Erlend Øien's master's thesis submitted to NTNU for the course TDT4900, amounting to 30 credits.

## Reproduce
**NB**: Some steps might rely on different directory names or paths depending on your project structure,
specifically related to data.
### Environment
- `merlin-pytorch`-container
- Other packages necessary:
    - Preprocessing: `multiprocesspandas`, `piso`,
    - EDA: `matplotlib`, `seaborn`, `statsmodels`
    - hyperparameter search: `optuna`

### Preprocessing
#### Ednet
1. Download dataset and merge the user files, or keep them separate, from TODO: Add link to dataset
2. Follow Notebook Ednet-01-Preliminary to retrieve lecture actions
3. Run `array_ednet_create_sessions.py` for each individual partition - Make sure it is partitioned by `user_id`
4. Run `array_ednet_feature_extract.py` for each individual partition, with the outputted partitions of step 3
5. Follow the procedure in notebook `ednet-05-dataset-creation.ipynb` to normalize, reduce and split the dataset
6. Create the baseline dataset, a utility matric using `create_repetitions_dataset.ipynb`

#### MOOCCubeX
1. Download dataset and merge the user files, or keep them separate, update paths in scripts accordingly.
3. Run `array_mooc_create_sessions.py` for each individual partition - Make sure it is partitioned by `user_id`
4. Run `array_mooc_feature_extract.py` for each individual partition, with the outputted partitions of step 3
4. Use the steps provided in `mooc-03-text_embedding.ipynb` for creating the pretrained-embeddings.
5. Follow the procedure in notebook `mooc-04-dataset-creation.ipynb` to normalize, reduce and split the dataset
6. Create the baseline dataset, a utility matric using `create_repetitions_dataset.ipynb`

### Hyperparameter tuning
- The search for parameters are found in the `configs`-directory and the respective search spaces.
- The optimally found hyperparameters for reproduction are found in the `hyperparameter`-directory.

#### Run hyperparameter tuning
- For Hyperparameter tuning of the SARS: use `hp_search.py`
    - Example usage: 
```bash
python hp_search.py \
    -d ednet -m sequential -f bias_adj_all_scaled \
    -l fix/"$BASE_PATH"_hp.log \
    -o "$BASE_PATH"_trials \
    --study-name ednet_xlnet_bias_adj \
    --model-type xlnet \
    --continuous-projection \
    --aggregated-projection \
    -C tags \
    -F num_forward num_backward num_pause median_pause time_spent seg_rep_60 \
        time_comp time_played replay_length skipped_length 
```
- For Hyperparameter tuning of the CF baselines, use `hp_baselines.py`
    - Example usage: 
```bash
python hp_baselines.py \
    -d mooc -m conventional -f all_scaled \
    -l fix/"$BASE_PATH"_hp.log \
    -o "$BASE_PATH"_trials \
    --study-name "$BASE_PATH" \
    --model-type knn \
    --num-trials-completed 0 \
```

- For HP tuning of repetition prediction models, use `hp_rep_classification.py`
    - Example usage: 
```bash
python hp_rep_classification.py \
    -d ednet -m conventional -f all_scaled \
    -l fix/"$BASE_PATH"_hp.log \
    -o "$BASE_PATH" \
    --study-name ednet_xgboost \
    --model-type xgboost \
    --num-trials-completed 168 \
    -F num_forward num_backward num_pause median_pause time_spent seg_rep_60 \
        time_comp time_played replay_length skipped_length
```

### Evaluation
- The specific configurations are given by the `hyperparameter`-directory, one must only load the correct `.json`-file

#### Sars Evaluation
- Example usage: 
```bash
# Projection flags are only for documentation as they are read from HP config
python evaluate_model.py \
    -d mooc -f all_scaled -m sequential \
    -p hyperparameters/mooc_gru_full.json \
    -l "$BASE_PATH".log \
    -o "$BASE_PATH" \
    --model-type gru \
    --aggregated-projection \
    --num-trials-completed 0 \
    -C fields concepts \
    -F num_forward num_backward num_pause median_pause time_spent seg_rep_60 \
        time_comp time_played replay_length skipped_length \
        avg_speed std_speed eff_speed

```

#### Conventional Baselines
- Use `eval_baselines.py` to evaluate the CF baselines
- Example usage: 
```bash
python eval_baselines.py \
    -d mooc -m conventional -f all_scaled \
    -l fix/"$BASE_PATH".log \
    -o "$BASE_PATH" \
    --hyperparameter-path hyperparameters/mooc_bpr.json\
    --model-type bpr \
    --num-trials-completed 0 \
```

#### Repetition prediction
- Run the classification models using `eval_rep_classification.py`
- Example usage: 
```bash
python eval_rep_classification.py \
    -d mooc -m conventional -f all_scaled \
    -l fix/"$BASE_PATH"_eval_full.log \
    -o "$BASE_PATH" \
    -p hyperparameters/mooc_log.json \
    --model-type log_loss \
    -F num_forward num_backward num_pause median_pause time_spent seg_rep_60 \
        time_comp time_played replay_length skipped_length\
        avg_speed std_speed eff_speed
```

#### Repetition alignment
- For retrieving the repetition alignment and ranking metrics, use `get_repetition_metrics.py`
- Example usage: 
```bash
    python get_repetition_metrics.py  \
    -d mooc -f all_scaled -m conventional \
    -l fix/"$BASE_PATH".log \
    --out-dir repetition_results \
```


### Naive baselines
- Simply run the steps in notebook `Non-personalised-baselines.ipynb`



