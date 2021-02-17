from math import e
import numpy as np
import click
import wandb

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def train_model():
    return np.random.randn(10,10)

@click.command()
@click.option("--epochs",default=10)
@click.option("--learning_rate",default=1e-4)
@click.option("--tags", multiple=True, default=None, help="List of tags to save to wandb")
def main(epochs=10,
         learning_rate=1e-4,
         tags = None):
    
    # Init WandB
    run = wandb.init(project="demo", entity="mikeamerrill",
                    tags=tags) 
    run.save()

    logger.info("Starting Training...")
 
    loss_gen = (e**(-1*x) * np.random.rand() for x in range(epochs))
    for epoch in range(epochs):
        weights = train_model()
        loss = next(loss_gen)
        wandb.log({"Loss": loss, "Epoch":epoch})

        # Save model to wandb
        np.save("weights", weights)
        wandb.save("weights.npy")
    
    logging.info("Ending Training...")

if __name__ == "__main__":
    main()

