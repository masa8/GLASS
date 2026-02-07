from datetime import datetime
from pathlib import Path

import pandas as pd
import os
import logging
import sys
import click
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_PARENT = PROJECT_ROOT.parent

for path in (PROJECT_ROOT, PROJECT_PARENT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from commands.net import utils  # noqa: E402

from commands.dataset import dataset  # noqa: E402
from commands.net import net  # noqa: E402


@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="ckpt")
def main(**kwargs):
    pass


main.add_command(net)
main.add_command(dataset)


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    result_collect = []
    data = {"Class": [], "Distribution": [], "Foreground": []}
    df = pd.DataFrame(data)
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name
        imagesize = dataloaders["training"].dataset.imagesize
        glass_list = methods["get_glass"](imagesize, device)

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, GLASS in enumerate(glass_list):
            flag = 0.0, 0.0, 0.0, 0.0, 0.0, -1.0
            if GLASS.backbone.seed is not None:
                utils.fix_seeds(GLASS.backbone.seed, device)

            GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
            if test == "ckpt":
                flag = GLASS.trainer(
                    dataloaders["training"], dataloaders["testing"], dataset_name
                )
                if isinstance(flag, int):
                    row_dist = {
                        "Class": dataloaders["training"].name,
                        "Distribution": flag,
                        "Foreground": flag,
                    }
                    df = pd.concat([df, pd.DataFrame(row_dist, index=[0])])

            if not isinstance(flag, int):
                i_auroc, i_ap, p_auroc, p_ap, p_pro, epoch = GLASS.tester(
                    dataloaders["testing"], dataset_name
                )
                result_collect.append(
                    {
                        "dataset_name": dataset_name,
                        "image_auroc": i_auroc,
                        "image_ap": i_ap,
                        "pixel_auroc": p_auroc,
                        "pixel_ap": p_ap,
                        "pixel_pro": p_pro,
                        "best_epoch": epoch,
                    }
                )

                if epoch > -1:
                    for key, item in result_collect[-1].items():
                        if isinstance(item, str):
                            continue
                        elif isinstance(item, int):
                            print(f"{key}:{item}")
                        else:
                            print(f"{key}:{round(item * 100, 2)} ", end="")

                # save results csv after each category
                print("\n")
                result_metric_names = list(result_collect[-1].keys())[1:]
                result_dataset_names = [
                    results["dataset_name"] for results in result_collect
                ]
                result_scores = [
                    list(results.values())[1:] for results in result_collect
                ]
                utils.compute_and_store_final_results(
                    run_save_path,
                    result_scores,
                    result_metric_names,
                    row_names=result_dataset_names,
                )

    # save distribution judgment xlsx after all categories
    if len(df["Class"]) != 0:
        os.makedirs("./datasets/excel", exist_ok=True)
        xlsx_path = (
            "./datasets/excel/" + dataset_name.split("_")[0] + "_distribution.xlsx"
        )
        df.to_excel(xlsx_path, index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
