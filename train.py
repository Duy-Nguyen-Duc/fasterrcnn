from torch_utils.engine import (
    train_one_epoch, evaluate
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from utils.general import (
    set_training_dir, Averager, 
    save_model_state, save_train_loss_plot,
    show_tranformed_image
)

import torch
import argparse
import yaml

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='fasterrcnn_resnet50',
        help='name of the model'
    )
    parser.add_argument(
        '-c', '--config', default=None,
        help='path to the data config file'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-e', '--epochs', default=5, type=int,
        help='number of epochs to train for'
    )
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch-size', dest='batch_size', default=8, type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-ims', '--img-size', dest='img_size', default=512, type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-pn', '--project-name', default=None, type=str, dest='project_name',
        help='training result dir name in outputs/training/, (default res_#)'
    )
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)
    
    # Settings/parameters/constants.
    TRAIN_DIR_IMAGES = data_configs['TRAIN_DIR_IMAGES']
    TRAIN_DIR_LABELS = data_configs['TRAIN_DIR_LABELS']
    VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
    VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    NUM_EPOCHS = args['epochs']
    SAVE_VALID_PREDICTIONS = data_configs['SAVE_VALID_PREDICTION_IMAGES']
    BATCH_SIZE = args['batch_size']
    VISUALIZE_TRANSFORMED_IMAGES = data_configs['VISUALIZE_TRANSFORMED_IMAGES']
    OUT_DIR = set_training_dir(args['project_name'])

    # Model configurations
    IMAGE_WIDTH = args['img_size']
    IMAGE_HEIGHT = args['img_size']
    device = 'cuda:0'
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS,
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, VALID_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader, DEVICE, CLASSES)

    # Initialize the Averager class.
    train_loss_hist = Averager()
    # Train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations.
    train_loss_list = []

    create_model = create_model[args['model']]
    model = create_model(num_classes=NUM_CLASSES)
    print(model)
    model = model.to(DEVICE)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]
    # Define the optimizer.
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)
    # LR will be zero as we approach `steps` number of epochs each time.
    # If `steps = 5`, LR will slowly reduce to zero every 5 epochs.
    steps = NUM_EPOCHS + 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=steps,
        T_mult=1,
        verbose=False
    )

    for epoch in range(NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
            scheduler=None
        )

        evaluate(
            model, 
            valid_loader, 
            device=DEVICE,
            save_valid_preds=SAVE_VALID_PREDICTIONS,
            out_dir=OUT_DIR,
            classes=CLASSES
        )

        # Add the current epoch's batch-wise lossed to the `train_loss_list`.
        train_loss_list.extend(batch_loss_list)

        # Save the current epoch model state. This can be used 
        # to resume training. It saves model state dict, number of
        # epochs trained for, optimizer state dict, and loss function.
        save_model_state(epoch, model, optimizer, OUT_DIR)

        # Save loss plot.
        save_train_loss_plot(OUT_DIR, train_loss_list)