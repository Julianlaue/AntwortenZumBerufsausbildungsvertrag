import click
import torch

from src.model_download.falcon_downloader import start_download


@click.group()
def cli():
    pass


@cli.command()
@click.option("--version",
              required=True,
              help="What version of the falcon model from Huggingface should be downloaded. \n Available are "
                   "'7b', '7b-inst','40b', '40b-inst'")
@click.option("--path",
              required=True,
              help="A path to the directory where the model files will be saved")
def download(version: str, path:str):
    """
    Starts downloading a falcon model from Huggingface: https://huggingface.co/tiiuae
    Example use on Neumann:  srun --gpus=1 --mem=32 python cli.py download --version=7b-inst --path=/home/julian.laue/models/falcon/falcon7b-inst
    """
    start_download(version, path)

if __name__ == '__main__':
    # Entrypoint for the training
    torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    cli()
