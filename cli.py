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
def download(model_version: str, path:str):
    """Starts downloading a falcon model from Huggingface. https://huggingface.co/tiiuae"""
    print('in cli download')
    start_download(model_version, path)

if __name__ == '__main__':
    # Entrypoint for the training
    torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    print('in main')
    cli()
