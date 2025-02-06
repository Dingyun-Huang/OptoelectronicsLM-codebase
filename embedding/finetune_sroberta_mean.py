"""Training script for fine-tuning the Sentence-Roberta model with mean pooling."""
from sentence_transformers import SentenceTransformer
import datasets
from accelerate import Accelerator
import accelerate.utils as accel_utils
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler, RobertaTokenizer, BertTokenizer, AlbertTokenizer, AutoTokenizer
from sbert import SentenceRoberta, GteLoss, InfoNCELoss, SentenceBert, SentenceAlbert
from torch.optim.lr_scheduler import ReduceLROnPlateau, ChainedScheduler


MODEL_NAME = "Dingyun-Huang/oe-albert-base-v2-mlm"
DATASET_NAME = "Dingyun-Huang/optoelectronics-title-abstract-triplets"
TRAINING_CONFIG = "configs/sroberta_mean.yaml"
ACCELERATE_CONFIG = 'configs/accelerate_config.yaml'

def load_config():
    """Load the training configuration from yaml file.

    Returns
    -------
    dict
        A dictionary containing the training configuration.
    """
    with open(TRAINING_CONFIG, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)
    return config

def load_accelerate_config():
    """Load the accelerator configuration from yaml file.

    Returns
    -------
    dict
        A dictionary containing the accelerator configuration.
    """
    with open(ACCELERATE_CONFIG, "r", encoding="utf8") as file:
        config = yaml.safe_load(file)
    return config


def get_optimizer(
    model: torch.nn.Module | SentenceTransformer,
    optimizer_config: dict,
):
    """
    Get the optimizer for the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to optimize.
    optimizer_config : dict
        A dictionary containing the optimizer configuration.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer for the model.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_config,
    )
    return optimizer


def get_dataloader(data_config: dict, split='train'):
    """
    Get the training dataloader.

    Parameters
    ----------
    data_config : dict
        A dictionary containing the data loading configuration.

    Returns
    -------
    torch.utils.data.DataLoader
        The training dataloader.
    """
    ds = datasets.load_dataset(DATASET_NAME)
    ds = ds.remove_columns(["negative", "doi"])
    return DataLoader(ds[split], **data_config[split])


def preprocess(batch, tokenizer, device):
    """
    Preprocess the batch.

    Parameters
    ----------
    batch : dict
        A dictionary containing the batch data.

    Returns
    -------
    dict
        A dictionary containing the preprocessed batch data.
    """
    anchor_inputs = tokenizer(
        batch['anchor'],
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=64
    ).to(device)

    positive_inputs = tokenizer(
        batch['positive'],
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    ).to(device)

    return anchor_inputs, positive_inputs


def evaluate(model, dataloader, tokenizer, device):
    """
    Evaluate the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for evaluation.

    Returns
    -------
    float
        The evaluation loss.
    """
    model.eval()
    total_loss = 0
    loss_func = InfoNCELoss()
    with torch.no_grad():
        for batch in dataloader:
            anchor_inputs, positive_inputs = preprocess(
                batch, tokenizer, device)
            anchor_outputs = model(**anchor_inputs)
            positive_outputs = model(**positive_inputs)
            loss = loss_func(anchor_outputs, positive_outputs)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def setup_scheduler(optimizer, config, training_dataloader):
    """
    Set up the scheduler for the optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to be used.
    config : dict
        The configuration dictionary containing the scheduler parameters.
    training_dataloader : torch.utils.data.DataLoader
        The training dataloader.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler
        The scheduler for the optimizer.
    """
    steps = int(len(training_dataloader) * config['epochs'])
    warmup_steps = int(steps * 0.1)
    scheduler_config = {'name': config['scheduler']['name'],
                        'num_training_steps': steps, 'num_warmup_steps': warmup_steps}
    config['scheduler'] = scheduler_config
    linear_scheduler = get_scheduler(optimizer=optimizer, **scheduler_config)
    # plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)
    # scheduler = ChainedScheduler([linear_scheduler, plateau_scheduler])
    return linear_scheduler, config


def main():
    """
    Main function for training the Sentence-Roberta model with mean pooling.
    """

    # Initialize the accelerator
    accelerator = Accelerator(log_with='wandb')
    device = accelerator.device
    config = load_config()
    config.update({'Accelerate': load_accelerate_config()})

    # Load the sentence-transformer model
    model = SentenceAlbert(
        base_model=MODEL_NAME,
        pooling_mode="mean",
    )

    for param in model.parameters():
        param.requires_grad = False

    # for param in model.pooler.parameters():
    #     param.requires_grad = True
    #     config['requires_grad'] = 'pooler'
    #     config['pooler'] = 'dense & tanh & mean pooling'

    for param in model.parameters():
        param.requires_grad = True
        config['requires_grad'] = 'all'
        config['pooler'] = 'linear dense & tanh & mean pooling'

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    # Initialize the loss function, optimizer, scheduler, and training dataloader
    loss_func = InfoNCELoss()
    config.update({'loss function': loss_func.__class__.__name__})
    optimizer = get_optimizer(model, config['optimizer'])
    training_dataloader = get_dataloader(config['data'], 'train')
    test_dataloader = get_dataloader(config['data'], 'test')

    # set up scheduler
    scheduler, config = setup_scheduler(optimizer, config, training_dataloader)

    # Prepare the model, optimizer, training dataloader, and scheduler
    model, optimizer, training_dataloader, scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, scheduler
    )

    # initialize the trackers
    accelerator.init_trackers(
        'fine-tune salbert',
        config=config,
        init_kwargs={
            'wandb': {
                'group': 'mean pooling',
            }
        }
    )

    step = 0
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}")
        for batch in tqdm(training_dataloader, disable=not accelerator.is_local_main_process):
            optimizer.zero_grad()
            anchor_inputs, positive_inputs = preprocess(
                batch, tokenizer, device)
            anchor_outputs = model(**anchor_inputs)
            positive_outputs = model(**positive_inputs)
            if config['gather']:

                gathered_anchor_outputs = accelerator.gather(anchor_outputs)
                gathered_positive_outputs = accelerator.gather(
                    positive_outputs)
                loss = torch.tensor(0.0).to(device)
                if accelerator.is_main_process:
                    loss = loss_func(gathered_anchor_outputs,
                                    gathered_positive_outputs)
                    accelerator.backward(loss)

                _ = accel_utils.broadcast(loss, 0)
                optimizer.step()
                scheduler.step()
            else:
                loss = loss_func(anchor_outputs, positive_outputs)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
            step += 1

            if step % 10 == 0 and accelerator.is_local_main_process:
                accelerator.log({'training_loss': loss.item()}, step=step)
                print(f"Step: {step}, Loss: {loss.item()}")
                accelerator.log(
                    {'learning_rate': scheduler.get_last_lr()[0]}, step=step)

            if step % 500 == 0:
                # continue
                print('-'*50)
                print("Evaluating the model...")
                test_loss = evaluate(model, test_dataloader, tokenizer, device)
                accelerator.log({'test_loss': test_loss}, step=step)
                print(f"Test Loss: {test_loss}")
                print('-'*50)
            
            if step % 1000 == 0:
                with accelerator.autocast():
                    accelerator.save_state(f"models/{model.__class__.__name__}/{loss_func.__class__.__name__}/2024-06-02-mean/checkpoint.{step}")

        # training_dataloader = get_dataloader(config['data'], 'train')
    accelerator.save_state("models/{model.__class__.__name__}/{loss_func.__class__.__name__}/final")


if __name__ == "__main__":
    main()
