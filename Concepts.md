
### Augement Parser

The augement parser is a built-in feature in Python that let's built CLI programs.

    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--devices", type=int, default=2)

    # Hyperparameters for the model
    parser.add_argument("--layer_1_dim", type=int, default=128)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    # Use the parsed arguments in your program
    trainer = Trainer(devices=args.devices)
    model = MyModel(layer_1_dim=args.layer_1_dim)

    

    python trainer.py --layer_1_dim 64 --devices 1


### Debugging 

- Set a breakpoint 
 - A breakpoint stops your code execution so you can inspect variables, and allow your code to execute one line at a time
 - The fast_dev_run argument in the trainer runs 5 batch of training, validation, text and prediction data through your trainer to see if there are any bugs
 - Sometimes it's helpful to only use a fraction of you training, val test, or predict data. 20% of training and 1% of the validation set. (limit_train_batches, limit_val_batches)
 - To add the child modules add a ModelSummary

### Bottlenecks

 - Profiling helps to find bottlenecks in the code by capturing analytics such as how long a function takes or how much memory is used
 - The simple profiler measures all the standard methods used in the training loop automatically
 - Another helpful technique to detect bottlenecks is to ensure that you are using the full capacity of your accelerator. DeviceStatsMonitor

