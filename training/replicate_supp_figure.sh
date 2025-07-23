for fold_id in 0 2 6
do
    python3 src/train.py --config=cfgs/unet/res18_monotemporal.yaml \
                     --trainer=cfgs/trainer_single_gpu.yaml \
                     --data=cfgs/data_monotemporal_full_features.yaml \
                     --seed_everything=0 \
                     --trainer.max_epochs=200 \
                     --do_test=True \
                     --data.data_fold_id=$fold_id
done