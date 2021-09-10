import argparse
import os
import shutil
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

from modules.utils import read_config
from modules.datasets import PointDataModule
from modules.models import PointNetModel

@rank_zero_only
def copy_checkpoint(src_checkpoint_path, dst_checkpoint_path):
    dst_checkpoint_path.parents[0].mkdir(parents=True, exist_ok=True)
    shutil.copy(src_checkpoint_path, dst_checkpoint_path)

def main(args):
    config = read_config(args.config_path)
    DATA_PATH = args.data_path
    OUT_PATH = args.out_path
    OUTPUT_CHECKPOINT = OUT_PATH + 'best_model.ckpt'
    LOG_DIR = 'work_dir/lightning_logs'
    config['num_workers'] = args.num_workers

    pl.seed_everything(config["seed"])

    dm = PointDataModule(
        DATA_PATH,
        valid_size=config['val_size'],
        config=config
    )

    dm.setup('fit')
    #ds = next(iter(dm.test_dataloader()))

    model = PointNetModel(config)
    tb_logger = TensorBoardLogger(LOG_DIR, name='name')
    checkpoint_path = os.path.join(tb_logger.log_dir, 'checkpoints')

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_last=True,
        **config['checkpoint_callback']
    )

    trainer = pl.Trainer(
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        gpus=[0],
        max_epochs=config['n_epochs']
    )
    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=dm)

    # Copy best checkpoint to best infer model folder
    checkpoint_path = checkpoint_callback.best_model_path
    if OUTPUT_CHECKPOINT is not None:
        copy_checkpoint(
            checkpoint_path,
            Path(OUTPUT_CHECKPOINT)
        )


def get_args():
    parser = argparse.ArgumentParser(
        description="Train model."
    )

    parser.add_argument(
        '-o', '--out-path',
        type=str, required=False,
        default='/home/bulat/ML/Kaggle/AICrowd/LidarCarDetection/PointNet/output/',
        help='Root directory of dataset.'
    )

    parser.add_argument(
        '-d', '--data-path',
        type=str, required=False,
        default='/home/bulat/ML/Kaggle/AICrowd/LidarCarDetection/input/',
        help='Root directory of dataset.'
    )
    parser.add_argument(
        "-c", "--config-path",
        type=str, required=False,
        default='./configs/train.yaml',
        help="Path to config."
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int, default=8,
        help=(
            "How many subprocesses to use for data loading. "
            "0 means that the data will be loaded in the main process."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
