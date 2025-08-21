import os
import shutil
import numpy as np
from argparse import ArgumentParser
from omegaconf import OmegaConf
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pradot.utils.metrics import compute_and_save_auroc

def ensemble_experiments(save_dir, exp_paths, compress=False):

    assert all(os.path.exists(exp) for exp in exp_paths), "Some experiment paths do not exist"
    configs = [OmegaConf.load(f"{exp}/.hydra/config.yaml") for exp in exp_paths]

    assert all(cfg.data.obj == configs[0].data.obj for cfg in configs), "Objects are not the same for all configs"

    all_preds = [np.load(f"{exp}/eval/predictions.npz", allow_pickle=True) for exp in exp_paths]
    assert all(pred.keys() == all_preds[0].keys() for pred in all_preds), "Keys are not the same shape for all configs"

    # copy all configs
    os.makedirs(os.path.join(save_dir, 'configs'), exist_ok=True)
    for exp in exp_paths:
        shutil.copyfile(
            os.path.join(exp, '.hydra', 'config.yaml'),
            os.path.join(save_dir, 'configs', os.path.basename(exp)+'.yaml')
        )

    # copy the image and metrics frolm base experiments
    for exp in exp_paths:
        if os.path.exists(os.path.join(exp, 'eval', 'predictions')):
            shutil.copytree(
                os.path.join(exp, 'eval', 'predictions'),
                os.path.join(save_dir, f'predictions_{os.path.basename(exp)}'),
                dirs_exist_ok=True
            )
        if os.path.exists(os.path.join(exp, 'eval', 'metrics.json')):
            shutil.copyfile(
                os.path.join(exp, 'eval', 'metrics.json'),
                os.path.join(save_dir, f'metrics_{os.path.basename(exp)}.json')
            )
    
    is_mvtecad = True if configs[0].project_name == "pradot_mvtecad" else False
    pixel_gt = [] if is_mvtecad else None
    pixel_pred = [] if is_mvtecad else None

    # copy dataset to run evaluation
    data_path = configs[0].data.data_path
    obj = configs[0].data.obj
    for defect in sorted(os.listdir(os.path.join(data_path, obj, 'test'))):
        if defect != 'good':
            os.makedirs(os.path.join(save_dir, 'copy_dataset', obj, 'ground_truth', defect), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'pred_for_eval', obj, 'test', defect), exist_ok=True)
    if os.path.exists(os.path.join(data_path, obj, 'defects_config.json')):
        shutil.copyfile(os.path.join(data_path, obj, 'defects_config.json'),
                        os.path.join(save_dir, 'copy_dataset', obj, 'defects_config.json'))
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    # go through all predictions
    for k in all_preds[0].keys():
        preds_k = [p[k][()] for p in all_preds]
        type_ = preds_k[0]['infos']['type'][0]
        img_id = k

        if is_mvtecad:
            os.makedirs(os.path.join(
                    save_dir, 'copy_dataset', obj, 'test', type_
            ), exist_ok=True)
            os.makedirs(os.path.join(
                save_dir, 'pred_for_eval', obj, 'test', type_
            ), exist_ok=True)

            img = preds_k[0]['image'].cpu().numpy()
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img, mode='RGB')
            img.save(os.path.join(
                save_dir, 'copy_dataset', obj, 'test', type_, f"{img_id}.png"
            ))

            if type_ != 'good':
                os.makedirs(os.path.join(
                    save_dir, 'copy_dataset', obj, 'ground_truth', type_
                ), exist_ok=True)

                gt = preds_k[0]['gt']
                gt = np.clip(gt, 0, 1)
                gt = (gt * 255).astype(np.uint8)
                gt_img = Image.fromarray(gt, mode='L')
                gt_img.save(os.path.join(
                    save_dir, 'copy_dataset', obj, 'ground_truth',
                        type_, f"{img_id}_mask.png"
                    ))
        else:

            if preds_k[0]['infos']['type'][0] != 'good':
                os.makedirs(os.path.join(
                    save_dir, 'copy_dataset', obj, 'ground_truth', type_, img_id
                ), exist_ok=True)

                for i in range(len(preds_k[0]['infos']['gt_id'][0])):
                    gt = preds_k[0]['gt']
                    gt = np.clip(gt, 0, 1)
                    gt = (gt * 255).astype(np.uint8)
                    gt_img = Image.fromarray(gt, mode='L')
                    gt_img.save(os.path.join(
                        save_dir, 'copy_dataset', obj, 'ground_truth', type_,
                        img_id, f"{preds_k[0]['infos']['gt_id'][0][i]}.png"
                    ))

        preds = [preds_k[i]['anomap'][0] for i in range(len(preds_k))]

        pred = np.mean(preds, axis=0)   
        
        if is_mvtecad:
            pixel_pred.append(pred.flatten())
            pixel_gt.append(preds_k[0]['gt'].flatten())

        img = Image.fromarray(pred)
        img.save(os.path.join(
            save_dir, 'pred_for_eval', obj, 'test', type_, f"{img_id}.tiff"
        ))

        # save the prediction
        fig, ax = plt.subplots()
        ax.set_title(f'Anomaly map for image {img_id}')
        ax.axis('off')
        ax.imshow(np.transpose(preds_k[0]['image'], (1, 2, 0)))
        ax.imshow(pred, alpha=0.7)
        ax.imshow(preds_k[0]['gt'], cmap='gray')
        fig.savefig(os.path.join(save_dir, 'predictions', f'predictions_{img_id}.png'))
        plt.close(fig)

    # compute the performance
    if is_mvtecad:
        os.system(f"python {os.path.join(data_path, 'mvtec_ad_evaluation', 'evaluate_experiment.py')} \
            --evaluated_objects {obj} \
            --dataset_base_dir {os.path.join(save_dir, 'copy_dataset')} \
            --anomaly_maps_dir {os.path.join(save_dir, 'pred_for_eval')} \
            --pro_integration_limit 0.05 \
            --output_dir {save_dir} \
            ")
        
        # compute pixel auroc
        pixel_pred = np.array(pixel_pred).flatten()
        pixel_gt = np.array(pixel_gt).flatten()
        order = np.argsort(pixel_pred)
        pixel_pred, pixel_gt = pixel_pred[order], pixel_gt[order]
        pixel_auroc = compute_and_save_auroc(
            pixel_gt, pixel_pred, path=None
        )

        with open(os.path.join(save_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)

        with open(os.path.join(save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            metrics['mean_localization_au_roc'] = pixel_auroc
            json.dump(metrics, f, ensure_ascii=False, indent=4)
    else:
        os.system(f"python {os.path.join(data_path, 'mvtec_loco_ad_evaluation', 'evaluate_experiment.py')} \
                --object_name {obj} \
                --dataset_base_dir {os.path.join(save_dir, 'copy_dataset')} \
                --anomaly_maps_dir {os.path.join(save_dir, 'pred_for_eval')} \
                --output_dir {save_dir} \
                ")

    if compress:
        # compress the images
        all_images = [
            Image.open(os.path.join(save_dir, 'predictions', img_n)).convert('RGB')
            for img_n in sorted(os.listdir(os.path.join(save_dir, 'predictions'))) if img_n.endswith('.png')
        ]
        all_images[0].save(
            os.path.join(save_dir, 'predictions', 'all_images.pdf'),
            "PDF", resolution=300.0, save_all=True, append_images=all_images[1:]
        )
        for img_n in os.listdir(os.path.join(save_dir, 'predictions')):
            if os.path.isfile(os.path.join(save_dir, 'predictions', img_n)) and not img_n.endswith('.pdf'):
                os.remove(os.path.join(save_dir, 'predictions', img_n))

    shutil.rmtree(os.path.join(save_dir, 'copy_dataset'))
    shutil.rmtree(os.path.join(save_dir, 'pred_for_eval'))

if __name__ == "__main__":
    aparser = ArgumentParser()
    aparser.add_argument('save_dir', type=str, help='Path to the save directory')
    aparser.add_argument('--exp_paths', type=str, nargs='+', help='Paths to the experiment directories')
    aparser.add_argument('--no_compress', action='store_true', help='Do not compress the images')

    args = aparser.parse_args()
    exps = [exp[:-1] if exp.endswith('/') else exp for exp in args.exp_paths]
    ensemble_experiments(args.save_dir, exps, not args.no_compress)

