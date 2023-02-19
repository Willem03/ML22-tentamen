from datetime import datetime

import torch
from loguru import logger

from tentamen.data import datasets
from tentamen.model import Accuracy
from tentamen.settings import presets
from tentamen.train import trainloop

if __name__ == "__main__":
    logger.add(presets.logdir / "01.log")

    trainstreamer, teststreamer = datasets.get_arabic(presets)

    from tentamen.model import AttentionGRU
    from tentamen.settings import AttentionGRUConfig

    configs_AttGRU = [
        AttentionGRUConfig(
            input_size=13,
            output_size=20,
            batch_size=951,
            hidden_size=255,
            embed_dim=255,
            tunedir=presets.logdir,
            num_layers=2,
            dropout=0.22,
            num_heads=4,
        ),
    ]

    for config in configs_AttGRU:
        model = AttentionGRU(config.dict())  # type ignore

        trainedmodel = trainloop(
            epochs=35,
            model=model,  # type: ignore
            optimizer=torch.optim.Adam,
            learning_rate=1e-3,
            loss_fn=torch.nn.CrossEntropyLoss(),
            metrics=[Accuracy()],
            train_dataloader=trainstreamer.stream(),
            test_dataloader=teststreamer.stream(),
            log_dir=presets.logdir,
            train_steps=len(trainstreamer),
            eval_steps=len(teststreamer),
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = presets.modeldir / (timestamp + presets.modelname)
    logger.info(f"save model to {path}")
    torch.save(trainedmodel, path)
