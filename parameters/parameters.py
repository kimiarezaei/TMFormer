import json


my_params = {
    "architecture": "CNN-transformer",
    "learning_rate": 1e-2,
    "warmup_period": 20,
    "eta_min": 1e-6,
    "batch_size": 1024,
    "epochs": 200,
    "random_seed": 42,
    "initialization": "he",
    "use_wandb": True,
    "scheduler": True,
    "apply_early_stop": True,
    "stop_threshold": 5,
    "kernel_size": 800,         
    "embedding_dim": 256,
    "stride": 700,  
    "num_heads": 8,
    "num_blks": 2,
    "drop_out_stoch": 0.3,
    "drop_out_att": 0.2,
    "weight_decay": 1e-5,
    "signal_length": 19200,
    "train_portion": 0.8,

}


with open('parameters/my_params.json', 'w') as json_file:
    json.dump(my_params, json_file, indent=4)

print("Dictionary has been saved as 'my_params.json'")