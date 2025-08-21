import os
import hydra
import pytorch_lightning
from omegaconf import OmegaConf

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pradot.utils import instantiate_callbacks, instantiate_loggers

@hydra.main(version_base=None, config_path="../config", config_name="config.yaml")
def train(cfg):

    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(OmegaConf.to_yaml(cfg, resolve=True))

    pytorch_lightning.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.data)

    model = hydra.utils.instantiate(cfg.model, data_params=cfg.data, optim_params=cfg.optimizer, _recursive_=False)

    logger = instantiate_loggers(cfg.logger)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.train:
        trainer.fit(train_dataloaders=datamodule, model=model)

    if cfg.test:
        trainer.test(model=model, dataloaders=datamodule, ckpt_path=trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    train()