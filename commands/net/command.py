import click

from . import backbones
from . import glass


@click.command("net")
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--meta_epochs", type=int, default=640)
@click.option("--eval_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--pre_proj", type=int, default=1)
@click.option("--mining", type=int, default=1)
@click.option("--noise", type=float, default=0.015)
@click.option("--radius", type=float, default=0.75)
@click.option("--p", type=float, default=0.5)
@click.option("--lr", type=float, default=0.0001)
@click.option("--svd", type=int, default=0)
@click.option("--step", type=int, default=20)
@click.option("--limit", type=int, default=392)
def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    meta_epochs,
    eval_epochs,
    dsc_layers,
    dsc_hidden,
    dsc_margin,
    train_backbone,
    pre_proj,
    mining,
    noise,
    radius,
    p,
    lr,
    svd,
    step,
    limit,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for _ in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_glass(input_shape, device):
        glasses = []
        for backbone_name, layers in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = (
                    backbone_name.split(".seed-")[0],
                    int(backbone_name.split("-")[-1]),
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            glass_inst = glass.GLASS(device)
            glass_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                mining=mining,
                noise=noise,
                radius=radius,
                p=p,
                lr=lr,
                svd=svd,
                step=step,
                limit=limit,
            )
            glasses.append(glass_inst.to(device))
        return glasses

    return "get_glass", get_glass
