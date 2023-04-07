import sys
from typing import NoReturn
import torch
import hydra
import os
import datetime
import logging
from timm.loss import LabelSmoothingCrossEntropy

from src.models.engine import train_epoch, valid_epoch
from src.models.swin_model import get_model
from src.utils.utils import (
    update_lr,
    get_dict_classes
)
from src.enities.train_pipeline import TrainingPipelineParams
from src.data.loaders import get_loaders

_log_format = "%(asctime)s\t%(levelname)s\t%(name)s\t" \
              "%(filename)s.%(funcName)s " \
              "line: %(lineno)d | \t%(message)s"
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(_log_format))
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def train_pipeline(params: TrainingPipelineParams) -> NoReturn:
    epochs = params.train_params.epochs
    lr = params.train_params.lr
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"The currently used device: {device}")
    logger.info(f"Num epochs: {epochs}")
    logger.info(f"Lr: {lr}")
    logger.info(f"Batch size: {params.train_params.batch_size}")
    logger.info(f"Image size: {params.train_params.img_size}")

    result_saving_directory = str(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M"))
    params.train_params.path_save_checkpoint = os.path.join(params.train_params.path_save_checkpoint,
                                                            result_saving_directory)
    os.makedirs(params.train_params.path_save_checkpoint, exist_ok=True)
    encode_classes2index, decode_index2classes = get_dict_classes(params.train_params.path_to_classes)
    model = get_model(params.train_params.name_model,
                      params.train_params.path_to_weights,
                      len(encode_classes2index),
                      params.train_params.pretrained).to(device)

    criterion = LabelSmoothingCrossEntropy()  # this is better than nn.CrossEntropyLoss
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)  # Setting for transfer learning
    train_data, train_loader, valid_data, valid_loader = get_loaders(params.train_params)
    logger.info(f"Training dataset size: {len(train_data)}")
    logger.info(f"Validating dataset size: {len(valid_data)}")
    curr_lr = lr
    best_f1_score = 0
    logger.info(f"--------==== Start of training ====--------")
    for epoch in range(epochs):

        train_epoch_loss, train_epoch_acc, train_epoch_f1_score = train_epoch(
            model, train_loader, optimizer, criterion, train_data, decode_index2classes, device
        )
        valid_epoch_loss, valid_epoch_acc, valid_epoch_f1_score = valid_epoch(
            model, valid_loader, criterion, valid_data, decode_index2classes, device
        )
        mean_f1_score_training = train_epoch_f1_score
        mean_f1_score_validating = valid_epoch_f1_score
        logger.info(f"Train Loss: {train_epoch_loss:.4f} ")
        logger.info(f"Train Accuracy: {train_epoch_acc:.4f}")
        logger.info(f"Train F1 Score MEAN: {mean_f1_score_training:.4f}")

        logger.info(f'Val Loss: {valid_epoch_loss:.4f}')
        logger.info(f'Val Accuracy: {valid_epoch_acc:.4f}')
        logger.info(f'Val F1 Score MEAD {mean_f1_score_validating:.4f}')

        if mean_f1_score_validating > max(best_f1_score, 0.5):
            best_f1_score = mean_f1_score_validating
            logger.info(f"New best score: {best_f1_score}")
            logger.info(f"Save checkpoint to {params.train_params.path_save_checkpoint}")
            with open(os.path.join(params.train_params.path_save_checkpoint,
                                   f'results_{params.train_params.name_model}.txt'), 'w') as f:
                f.write(f"Train Loss: {train_epoch_loss:.4f} "
                        f"Train Accuracy: {train_epoch_acc:.4f} "
                        f"Train F1 Score: {mean_f1_score_training:.4f}\n")
                f.write(f'Val Loss: {valid_epoch_loss:.4f} '
                        f'Val Accuracy: {valid_epoch_acc:.4f} '
                        f'Val F1 Score: {mean_f1_score_validating:.4f}')

            torch.save(model.state_dict(),
                       os.path.join(params.train_params.path_save_checkpoint,
                                    f"checkpoint_{params.train_params.name_model}.pth"))

        if (epoch + 1) % 2 == 0:
            curr_lr = curr_lr * 0.8
            update_lr(optimizer, curr_lr)


if __name__ == "__main__":
    train_pipeline()
