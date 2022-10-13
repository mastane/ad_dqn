import wandb
from random import random



for repeat in range(3):
  wandb.init(project="test", entity="kirill-fedyanin")
  wandb.config.update({
    "learning_rate": 0.001*repeat,
    "epochs": 100,
    "batch_size": 128
  })

  for loss in range(20):
    wandb.log({'loss': loss + random()})

  wandb.finish()
