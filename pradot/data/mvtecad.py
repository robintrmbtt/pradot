import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from lightning import LightningDataModule, Callback
from PIL import Image
import shutil
import json
from omegaconf import ListConfig
import itertools

from pradot.utils import compute_and_save_auroc
from pradot.utils.projection import get_features_and_coords_proj, proj2d

class ReadImage:
    def __init__(self, mode=torchvision.io.ImageReadMode.RGB, is_gt=False):
        self.mode = mode
        self.is_gt = is_gt

    def __call__(self, path):
        if not self.is_gt:
            x = torchvision.io.read_image(path, mode=self.mode).to(torch.float32)
            if x.shape[0] == 1:
                x = x.repeat(3,1,1)
            return x / 255.
        else:
            x = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.GRAY).to(torch.float32)
            x = torch.clamp(x, min=0, max=1.)
            return x

class CustomDataset(Dataset):
    def __init__(self, data, img_size, transforms=None):
        self.set_io_transforms(img_size)
        self.prepare_data(data)

        self.transforms = transforms
        
    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gt[index]
       
        if self.transforms is not None:
            image = self.transforms(image)
            if gt is not None:
                gt = self.transforms(gt)

        res = {'image': image}
        if gt is not None:
            res['gt'] = gt
        if self.infos[index] is not None:
            res['infos'] = self.infos[index]
        
        return res
    
    def __len__(self):
        return len(self.images)

    def set_io_transforms(self, img_size):
        self.io_transforms_img = transforms.Compose([
            ReadImage(),
            transforms.Resize(img_size)
        ])

        self.io_transforms_gt = transforms.Compose([
            ReadImage(is_gt=True),
            transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
        ])

    def prepare_data(self, data):
        def prep_gt(gt):
            if gt is not None:
                return self.io_transforms_gt(gt)
            return gt
        
        def get_info(data):
            if 'infos' in data:
                return data['infos']
            else:
                return None
        
        self.images = list(map(self.io_transforms_img, tqdm([data[k]['image'] for k in range(len(data))], desc='Processing images...')))
        self.gt = list(map(prep_gt, tqdm([data[k]['gt'] for k in range(len(data))], desc='Processing ground truth...')))
        self.infos = list(map(get_info, data))


def create_dataset_mvtec(data_path, obj, img_size, val_ratio=0.2, n_train=None, n_test=None, stages=['train', 'val', 'test']):
    data_test = []

    data_train = np.array([
        {'image': os.path.join(data_path, obj, 'train', 'good', filename), 
        'gt': None} 
        for filename in sorted(os.listdir(os.path.join(data_path, obj, 'train', 'good')))
    ])
    
    if n_train is not None:
        data_train = data_train[:n_train]

    np.random.shuffle(data_train)
    data_val = data_train[:int(val_ratio * len(data_train))]
    data_train = data_train[int(val_ratio * len(data_train)):]

    for dirn in sorted(os.listdir(os.path.join(data_path, obj, 'test'))):
        if dirn=='good':
            data_test += [(
                {'image': os.path.join(data_path, obj, 'test', dirn, filename), 
                'gt': None,
                'infos': {
                    'img_id': filename.split('.')[0],
                    'type': dirn,
                }}        
            ) for filename in sorted(os.listdir(os.path.join(data_path, obj, 'test', dirn)))]

        else:
            data_test += [(
                {'image': os.path.join(data_path, obj, 'test', dirn, filename), 
                'gt': os.path.join(data_path, obj, 'ground_truth', dirn, filename.split('.')[0]+'_mask.png'),
                'infos': {
                    'img_id': filename.split('.')[0],
                    'type': dirn
                }}
            ) for filename in sorted(os.listdir(os.path.join(data_path, obj, 'test', dirn)))]


    if n_test is not None:
        data_test = data_test[:n_test]

    data = {}
    if 'train' in stages:
        data['train'] = CustomDataset(data_train, img_size)
    if 'val' in stages:
        data['val'] = CustomDataset(data_val, img_size)
    if 'test' in stages:
        data['test'] = CustomDataset(data_test, img_size)
    return data


class MVTecDataModule(LightningDataModule):
    def __init__(self, data_path, obj, img_size, batch_size, val_ratio=0.2, n_train=None, n_test=None, stages=['train', 'val', 'test'], **kwargs):
        super().__init__()
        self.data_path = data_path
        self.obj = obj
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.n_train = n_train
        self.n_test = n_test
        self.stages = stages
        
    def setup(self, stage=None):
        self.data = create_dataset_mvtec(
            self.data_path, self.obj, self.img_size, 
            val_ratio=self.val_ratio,
            n_train=self.n_train,
            n_test=self.n_test,
            stages=self.stages
        )
        # self.data_train, self.data_val, self.data_test = create_dataset_mvtec(
        #     self.data_path, self.obj, self.img_size, 
        #     val_ratio=self.val_ratio,
        #     n_train=self.n_train,
        #     n_test=self.n_test
        # )

    def train_dataloader(self):
        return DataLoader(self.data['train'], batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.data['val'], batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.data['test'], batch_size=1, shuffle=False, num_workers=4)



############
## Evaluation callback
############


class EvalMVTecAD(Callback):
    def __init__(self, show_assignment, show_proto_recons, show_images_recons, projection_params, compress):
        if show_images_recons:
            assert show_proto_recons, 'To show reconstructed images, you need to show reconstructed prototypes as well.'
        self.results = {}
        self.show_assignment = show_assignment
        self.show_proto_recons = show_proto_recons
        self.show_images_recons = show_images_recons
        self.compress = compress      
        self.projection_params = projection_params

    def on_test_start(self, trainer, pl_module):
        self.save_dir = os.path.join(trainer.default_root_dir, 'eval')
        self.gt_and_pred = {
            'gts_pixel': [],
            'preds_pixel': [[] for _ in range(len(pl_module.model.out_indices)+int(len(pl_module.model.out_indices)>1))],
        }

        for defect in sorted(os.listdir(os.path.join(trainer.datamodule.data_path, trainer.datamodule.obj, 'test'))):
            if defect != 'good':
                os.makedirs(os.path.join(self.save_dir, 'copy_dataset', trainer.datamodule.obj, 'ground_truth', defect), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'pred_for_eval', trainer.datamodule.obj, 'test', defect), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, 'copy_dataset', trainer.datamodule.obj, 'test', defect), exist_ok=True)
        
        # os.makedirs(os.path.join(self.save_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'predictions'), exist_ok=True)
        if self.show_images_recons:
            os.makedirs(os.path.join(self.save_dir, 'images_recons'), exist_ok=True)
        if self.show_assignment:
            os.makedirs(os.path.join(self.save_dir, 'assignment'), exist_ok=True)
        if self.projection_params is not None:
            os.makedirs(os.path.join(self.save_dir, 'projection'), exist_ok=True)
            self.features_for_proj = {
                out_i: {
                    'features_abnormal': [],
                    'features_normal': [],
                    'coords_abnormal': [],
                    'coords_normal': []
                } for out_i in pl_module.model.out_indices
            }

        if self.show_proto_recons:
            os.makedirs(os.path.join(self.save_dir, 'proto_recons'), exist_ok=True)
            dl = torch.utils.data.DataLoader(
                trainer.datamodule.data['train'], 
                batch_size=1, 
                shuffle=False, 
                num_workers=4, 
                drop_last=False
            )
            pl_module.reconstruct_prototypes(os.path.join(self.save_dir, 'proto_recons'), dl=dl)
            

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self.results[batch_idx] = {
            'image': batch['image'][0].float().cpu().detach(),
            'gt': batch['gt'][0,0].cpu().numpy() if 'gt' in batch else np.zeros(pl_module.model.input_size),
            'infos': batch['infos'],
            'anomap': outputs['anomap'],
            'ot_dist': outputs['ot_dist'],
            'ot_plan': outputs['ot_plan']
        }

        #########
        ## save the image as tiff and gt to the copy dataset
        #########

        if batch['infos']['type'][0] != 'good':
            gt = batch['gt'][0][0].cpu().numpy()
            gt = np.clip(gt, 0, 1)
            gt = (gt * 255).astype(np.uint8)
            gt_img = Image.fromarray(gt, mode='L')
            gt_img.save(os.path.join(
                self.save_dir, 'copy_dataset', trainer.datamodule.obj, 'ground_truth',
                    batch['infos']['type'][0], f"{batch['infos']['img_id'][0]}_mask.png"
                ))
            
        img = batch['image'][0].cpu().numpy()
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img, mode='RGB')
        img.save(os.path.join(
            self.save_dir, 'copy_dataset', trainer.datamodule.obj, 'test',
            batch['infos']['type'][0], f"{batch['infos']['img_id'][0]}.png"
        ))
                    
        pred = outputs['anomap'].mean(0)
        pred = Image.fromarray(pred)
        pred.save(os.path.join(
            self.save_dir, 'pred_for_eval', trainer.datamodule.obj, 'test', 
            batch['infos']['type'][0], f"{batch['infos']['img_id'][0]}.tiff"
        ))

        self.gt_and_pred['gts_pixel'].append(self.results[batch_idx]['gt'].ravel())       

        if self.show_images_recons:
            for k in range(len(pl_module.model.out_indices)):
                pl_module.reconstruct_image(
                    outputs['ot_plan'][k],
                    k,
                    x=batch['image'],
                    save_path=os.path.join(
                        self.save_dir, 'images_recons', f'{batch_idx}_recons_scale_{pl_module.model.out_indices[k]}.png'
                    ),
                    title=f'Reconstructed image for image {batch_idx} scale {pl_module.model.out_indices[k]}'
                )
            

        if self.show_assignment:
            for k in range(len(pl_module.model.out_indices)):
                pl_module.visualize_assignments(outputs['ot_plan'][k], 
                                                    k, 
                                                    save_path=os.path.join(
                                                        self.save_dir, 'assignment', f'{batch_idx}_assignment_scale_{pl_module.model.out_indices[k]}.png'
                                                        ),
                                                    title=f'Assignment for image {batch_idx} scale {pl_module.model.out_indices[k]}'
                                                    )
        
        if self.projection_params is not None:
            features = pl_module.model(batch['image'])
            for k, out_i in enumerate(pl_module.model.out_indices):
                res = get_features_and_coords_proj(
                    features[k][0].detach().cpu(), 
                    torch.from_numpy(self.results[batch_idx]['gt']),
                    grid_coords=pl_module.model.grid_coords[k],
                    **self.projection_params
                )
                self.features_for_proj[out_i]['features_normal'].append(res['features_normal'])
                self.features_for_proj[out_i]['features_abnormal'].append(res['features_abnormal'])
                self.features_for_proj[out_i]['coords_normal'].append(res['coords_normal'])
                self.features_for_proj[out_i]['coords_abnormal'].append(res['coords_abnormal'])


    def on_test_end(self, trainer, pl_module):
        np.savez_compressed(os.path.join(self.save_dir, 'predictions.npz'), **{str(k):v for k,v in self.results.items()})

        ######
        ## SAVE ANOMALY MAPS
        ######

        print('Saving anomaly maps...')

        for idx, res in self.results.items():
            for k in range(len(res['anomap'])):
                self.gt_and_pred['preds_pixel'][k].append(res['anomap'][k].ravel())

                fig, axs = plt.subplots(1,2, figsize=(12,6))
                for ax in axs:
                    ax.axis('off')
                axs[0].set_title(f'Anomaly map for image {idx} scale {pl_module.model.out_indices[k]}')
                axs[0].imshow(np.transpose(res['image'],(1,2,0)))
                axs[0].imshow(res['anomap'][k], alpha=0.7)
                axs[1].imshow(res['gt'])
                axs[1].set_title('Ground Truth')
                plt.savefig(
                    os.path.join(self.save_dir, 'predictions', f'{idx}_scale_{pl_module.model.out_indices[k]}.png'), dpi=400
                    )
                plt.close(fig)

            if len(res['anomap']) > 1:
                # save pred for aggregated anomap
                anomap = np.mean(res['anomap'], axis=0)
                self.gt_and_pred['preds_pixel'][-1].append(anomap.ravel())

                fig, axs = plt.subplots(1,2, figsize=(12,6))
                for ax in axs:
                    ax.axis('off')
                axs[0].set_title('Anomaly map for image {batch_idx} aggregated')
                axs[0].imshow(np.transpose(res['image'],(1,2,0)))
                axs[0].imshow(anomap, alpha=0.7)
                axs[1].imshow(res['gt'])
                axs[1].set_title('Ground Truth')
                plt.savefig(os.path.join(self.save_dir, 'predictions', f'{idx}_aggregated.png'), dpi=400)
                plt.close(fig)


        ######
        ## COMPUTE METRICS
        ######
        print('Computing metrics...')
        os.system(f"python {os.path.join(trainer.datamodule.data_path, 'mvtec_ad_evaluation', 'evaluate_experiment.py')} \
            --evaluated_objects {trainer.datamodule.obj} \
            --dataset_base_dir {os.path.join(self.save_dir, 'copy_dataset')} \
            --anomaly_maps_dir {os.path.join(self.save_dir, 'pred_for_eval')} \
            --pro_integration_limit 0.05 \
            --output_dir {self.save_dir} \
            ")
        
        auroc_pixel = []
        n_scales = len(pl_module.model.out_indices)+int(len(pl_module.model.out_indices)>1)
        for k, scale in enumerate(list(pl_module.model.out_indices) + ['aggregated',] * (n_scales > 1)):
            pred = np.array(self.gt_and_pred['preds_pixel'][k]).flatten()
            order = np.argsort(pred)
            auroc = compute_and_save_auroc(
                np.array(self.gt_and_pred['gts_pixel']).flatten()[order],
                pred[order],
                path=None
            )
            auroc_pixel.append(auroc)
        
        # log metrics
        with open(os.path.join(self.save_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        if trainer.logger is not None:
            trainer.logger.log_metrics({
                'classif/auroc': metrics['mean_classification_au_roc'],
                'seg/auroc': auroc_pixel[-1],
                'seg/auspro5': metrics['mean_au_pro'],
            })

        with open(os.path.join(self.save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            metrics['mean_localization_au_roc'] = auroc_pixel[-1]
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        # remove unecessary folders
        shutil.rmtree(os.path.join(self.save_dir, 'copy_dataset'))
        shutil.rmtree(os.path.join(self.save_dir, 'pred_for_eval'))



        #######
        ## UMAP/TSNE PROJECTION
        ########

        if self.projection_params is not None:
            for k, out_i in enumerate(self.features_for_proj.keys()):
                self.features_for_proj[out_i]['features_normal'] = torch.concatenate(self.features_for_proj[out_i]['features_normal'], axis=0)
                self.features_for_proj[out_i]['features_abnormal'] = torch.concatenate(self.features_for_proj[out_i]['features_abnormal'], axis=0)
                self.features_for_proj[out_i]['coords_normal'] = torch.concatenate(self.features_for_proj[out_i]['coords_normal'], axis=0)
                self.features_for_proj[out_i]['coords_abnormal'] = torch.concatenate(self.features_for_proj[out_i]['coords_abnormal'], axis=0)
                if 'n_normal' in self.projection_params:
                    print(f"Subsampling normal features for scale {out_i} to {self.projection_params['n_normal']} (initially {self.features_for_proj[out_i]['features_normal'].shape[0]})")
                    inds = np.random.choice(
                        self.features_for_proj[out_i]['features_normal'].shape[0], 
                        min(self.projection_params['n_normal'], self.features_for_proj[out_i]['features_normal'].shape[0]), 
                        replace=False
                    )
                    self.features_for_proj[out_i]['features_normal'] = self.features_for_proj[out_i]['features_normal'][inds]
                    self.features_for_proj[out_i]['coords_normal'] = self.features_for_proj[out_i]['coords_normal'][inds]
                if 'n_abnormal' in self.projection_params:
                    print(f"Subsampling abnormal features for scale {out_i} to {self.projection_params['n_abnormal']} (initially {self.features_for_proj[out_i]['features_abnormal'].shape[0]})")
                    inds = np.random.choice(
                        self.features_for_proj[out_i]['features_abnormal'].shape[0], 
                        min(self.projection_params['n_abnormal'], self.features_for_proj[out_i]['features_abnormal'].shape[0]), 
                        replace=False
                    )
                    self.features_for_proj[out_i]['features_abnormal'] = self.features_for_proj[out_i]['features_abnormal'][inds]
                    self.features_for_proj[out_i]['coords_abnormal'] = self.features_for_proj[out_i]['coords_abnormal'][inds]
                if 'n_prototypes' in self.projection_params:
                    print(f"Subsampling prototypes for scale {out_i} to {self.projection_params['n_prototypes']} (initially {pl_module.model.prototypes_features[k].shape[0]})")
                    inds = np.random.choice(
                        pl_module.model.prototypes_features[k].shape[0], 
                        min(self.projection_params['n_prototypes'], pl_module.model.prototypes_features[k].shape[0]), 
                        replace=False
                    )
                    prototypes_for_proj = pl_module.model.prototypes_features[k].detach().cpu()[inds]
                    coords_prototypes_for_proj = pl_module.model.prototypes_coords[k].detach().cpu()[inds]
                else:
                    prototypes_for_proj = pl_module.model.prototypes_features[k].detach().cpu()
                    coords_prototypes_for_proj = pl_module.model.prototypes_coords[k].detach().cpu()
                    
                
                for proj_n, proj_params in self.projection_params['projections'].items():
                    
                    for key,v in proj_params.items():
                        if not isinstance(v, (list, ListConfig)):
                            proj_params[key] = [v]

                    for combination in itertools.product(*proj_params.values()):
                        d = dict(zip(proj_params.keys(), combination))
                        _ = proj2d(
                            features={
                                'abnormal': self.features_for_proj[out_i]['features_abnormal'],
                                'normal': self.features_for_proj[out_i]['features_normal'],
                                'prototypes': prototypes_for_proj
                            },
                            coords={
                                'abnormal': self.features_for_proj[out_i]['coords_abnormal'],
                                'normal': self.features_for_proj[out_i]['coords_normal'],
                                'prototypes': coords_prototypes_for_proj
                            },
                            proj_type=proj_n,
                            proj_params=d,
                            alpha=pl_module.alpha,
                            save_path=os.path.join(
                                self.save_dir, 
                                'projection', 
                                f"{proj_n}-scale_{out_i}-{'-'.join(['_'.join(map(str,i)) for i in d.items()])}-alpha_{pl_module.alpha}.png"
                            ),
                            title=f"{proj_n} scale {out_i} {' '.join(['='.join(map(str,i)) for i in d.items()])} alpha={pl_module.alpha}"
                        )


        #######
        ## COMPRESS FOLDERS IN ONE PDF EACH
        #######

        if self.compress:
            print('Compressing predictions...')
            all_images = [
            Image.open(os.path.join(self.save_dir, 'predictions', img_n)).convert('RGB')
                for img_n in sorted(os.listdir(os.path.join(self.save_dir, 'predictions')))
            ]
            all_images[0].save(
                os.path.join(self.save_dir, 'predictions', 'all_predictions.pdf'), 
                "PDF" , resolution=400.0, save_all=True, append_images=all_images[1:]
            )
            for img_n in os.listdir(os.path.join(self.save_dir, 'predictions')):
                if not img_n.endswith('.pdf'):
                    os.remove(os.path.join(self.save_dir, 'predictions', img_n))

            if self.show_images_recons:
                all_images = [
                    Image.open(os.path.join(self.save_dir, 'images_recons', img_n)).convert('RGB')
                    for img_n in sorted(os.listdir(os.path.join(self.save_dir, 'images_recons')))
                ]
                all_images[0].save(
                    os.path.join(self.save_dir, 'images_recons', 'all_images_recons.pdf'), 
                    "PDF" , resolution=400.0, save_all=True, append_images=all_images[1:]
                )
                for img_n in os.listdir(os.path.join(self.save_dir, 'images_recons')):
                    if not img_n.endswith('.pdf'):
                        os.remove(os.path.join(self.save_dir, 'images_recons', img_n))
                
            if self.show_assignment:
                all_images = [
                    Image.open(os.path.join(self.save_dir, 'assignment', img_n)).convert('RGB')
                    for img_n in sorted(os.listdir(os.path.join(self.save_dir, 'assignment')))
                ]
                all_images[0].save(
                    os.path.join(self.save_dir, 'assignment', 'all_assignments.pdf'), 
                    "PDF" , resolution=400.0, save_all=True, append_images=all_images[1:]
                )
                for img_n in os.listdir(os.path.join(self.save_dir, 'assignment')):
                    if not img_n.endswith('.pdf'):
                        os.remove(os.path.join(self.save_dir, 'assignment', img_n))


            if self.projection_params is not None:
                all_images = [
                    Image.open(os.path.join(self.save_dir, 'projection', img_n)).convert('RGB')
                    for img_n in sorted(os.listdir(os.path.join(self.save_dir, 'projection')))
                ]
                all_images[0].save(
                    os.path.join(self.save_dir, 'projection', 'all_projections.pdf'), 
                    "PDF" , resolution=400.0, save_all=True, append_images=all_images[1:]
                )
                for img_n in os.listdir(os.path.join(self.save_dir, 'projection')):
                    if not img_n.endswith('.pdf'):
                        os.remove(os.path.join(self.save_dir, 'projection', img_n))