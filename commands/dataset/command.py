import logging

import click
import torch

LOGGER = logging.getLogger(__name__)


@click.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.argument("aug_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=8, type=int, show_default=True)
@click.option("--num_workers", default=16, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--distribution", default=0, type=int)
@click.option("--mean", default=0.5, type=float)
@click.option("--std", default=0.1, type=float)
@click.option("--fg", default=1, type=int)
@click.option("--rand_aug", default=1, type=int)
@click.option("--downsampling", default=8, type=int)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    aug_path,
    subdatasets,
    batch_size,
    resize,
    imagesize,
    num_workers,
    rotate_degrees,
    translate,
    scale,
    brightness,
    contrast,
    saturation,
    gray,
    hflip,
    vflip,
    distribution,
    mean,
    std,
    fg,
    rand_aug,
    downsampling,
    augment,
):
    _DATASETS = {
        "mvtec": ["datasets.mvtec", "MVTecDataset"],
        "visa": ["datasets.visa", "VisADataset"],
        "mpdd": ["datasets.mvtec", "MVTecDataset"],
        "wfdd": ["datasets.mvtec", "MVTecDataset"],
    }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, test, get_name=name):
        dataloaders = []
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader.name = get_name + "_" + subdataset

            if test == "ckpt":
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
                    resize=resize,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.TRAIN,
                    seed=seed,
                    rotate_degrees=rotate_degrees,
                    translate=translate,
                    brightness_factor=brightness,
                    contrast_factor=contrast,
                    saturation_factor=saturation,
                    gray_p=gray,
                    h_flip_p=hflip,
                    v_flip_p=vflip,
                    scale=scale,
                    distribution=distribution,
                    mean=mean,
                    std=std,
                    fg=fg,
                    rand_aug=rand_aug,
                    downsampling=downsampling,
                    augment=augment,
                    batch_size=batch_size,
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=True,
                )

                train_dataloader.name = test_dataloader.name
                LOGGER.info(
                    f"Dataset {subdataset.upper():^20}: train={len(train_dataset)} test={len(test_dataset)}"
                )
            else:
                train_dataloader = test_dataloader
                LOGGER.info(
                    f"Dataset {subdataset.upper():^20}: train={0} test={len(test_dataset)}"
                )

            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return "get_dataloaders", get_dataloaders
