import os
import subprocess
import multiprocessing

# Define the base directory containing the runs
base_dir = '/home/sl221120/WildfireSpreadTS/lightning_logs/wildfire_progression'
data_dir = '/home/sl221120/scratch/WildfireSpreadTS_HDF5'

# Log data mapping run_id to data_fold_id
log_data = [
    {"run_id": "313hjm25", "data_fold_id": 0},
    {"run_id": "e5v3xgd0", "data_fold_id": 1},
    {"run_id": "4cjz4vbi", "data_fold_id": 2},
    {"run_id": "mn6grenj", "data_fold_id": 3},
    {"run_id": "4xb6m1v6", "data_fold_id": 4},
    {"run_id": "zaxbzcrw", "data_fold_id": 5},
    {"run_id": "8zbn9dje", "data_fold_id": 6},
    {"run_id": "b5c18545", "data_fold_id": 7},
    {"run_id": "fpofsvsv", "data_fold_id": 8},
    {"run_id": "vphrrk4v", "data_fold_id": 9},
    {"run_id": "pqq4e9kb", "data_fold_id": 10},
    {"run_id": "njyy633r", "data_fold_id": 11}
]


# Convert log data to a dictionary
run_id_to_fold_id = {entry["run_id"]: entry["data_fold_id"] for entry in log_data}

# Function to run the test command for a specific checkpoint
def run_test(checkpoint_path, data_fold_id, gpu_id):
    test_command = [
        'python',
        '/home/sl221120/WildfireSpreadTS/src/train.py',
        '--config=cfgs/unet/res18_monotemporal_test_only.yaml',
        '--trainer=cfgs/trainer_single_gpu.yaml',
        '--data=cfgs/data_monotemporal_full_features.yaml',
        '--seed_everything=0',
        '--trainer.max_epochs=200',
        '--do_test=True',
        f'--data.data_fold_id={data_fold_id}',
        f'--data.data_dir={data_dir}',
        '--data.features_to_keep=[0,1,2,3,4,5,6,7,8,9,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,38,39]',
        f'--ckpt_path={checkpoint_path}'
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Run the test command
    subprocess.run(test_command, env=env)

def process_run(run_id, data_fold_id, gpu_id):
    run_path = os.path.join(base_dir, run_id, 'checkpoints')
    checkpoint_file = os.path.join(run_path, os.listdir(run_path)[0])
    if checkpoint_file.endswith(".ckpt"):
        run_test(checkpoint_file, data_fold_id, gpu_id)

# Create a process for each run
processes = []
gpu_count = 4


for i, (run_id, data_fold_id) in enumerate(run_id_to_fold_id.items()):
    gpu_id = i % gpu_count
    p = multiprocessing.Process(target=process_run, args=(run_id, data_fold_id, gpu_id))
    processes.append(p)
    p.start()

# Wait for all processes to complete
for p in processes:
    p.join()
