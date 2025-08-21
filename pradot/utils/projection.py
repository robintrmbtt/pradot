import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import umap
from sklearn.manifold import TSNE

from pradot.utils import cosine_sim

def get_features_and_coords_proj(features, gt, normal_ratio=1., abnormal_ratio=1., gt_thr=0.2, grid_coords=None, **kwargs):
    c,h,w = features.shape
    features = features.reshape(c, -1).T
    H,W = gt.shape
    scale_h, scale_w = H // h, W // w

    # get grid coordinates
    if grid_coords is None:
        coords_grid_x = torch.linspace(0, 1, h)
        coords_grid_y = torch.linspace(0, 1, h)
        grid_coords = torch.stack(torch.meshgrid(coords_grid_x, coords_grid_y, indexing='ij')).permute(1, 2, 0).reshape(-1, 2)
        grid_coords = grid_coords
        
    patches_gt = gt.unfold(0, scale_h, scale_w).unfold(1, scale_h, scale_w)
    patches_gt = patches_gt.contiguous().view(-1, scale_h*scale_w)

    assert patches_gt.shape[0] == features.shape[0]

    is_abnormal = patches_gt.mean(dim=1) > gt_thr

    features_abnormal = features[is_abnormal]
    features_normal = features[~is_abnormal]

    grid_coords_abnormal = grid_coords[is_abnormal]
    grid_coords_normal = grid_coords[~is_abnormal]

    # random subsample features with parameters normal_ratio and abnormal_ratio
    ind_abnormal = np.random.choice(features_abnormal.shape[0], int(features_abnormal.shape[0] * abnormal_ratio), replace=False)
    ind_normal = np.random.choice(features_normal.shape[0], int(features_normal.shape[0] * normal_ratio), replace=False)
    features_abnormal = features_abnormal[ind_abnormal]
    features_normal = features_normal[ind_normal]
    grid_coords_abnormal = grid_coords_abnormal[ind_abnormal]
    grid_coords_normal = grid_coords_normal[ind_normal]

    return {
            'features_abnormal': features_abnormal.detach().cpu(),
            'features_normal': features_normal.detach().cpu(),
            'coords_abnormal': grid_coords_abnormal.detach().cpu(),
            'coords_normal': grid_coords_normal.detach().cpu(),
        }

def proj2d(features, coords, proj_type, proj_params, alpha, save_path=None, title=None):
    n_abnormal = features['abnormal'].shape[0]
    n_normal = features['normal'].shape[0]
    feat = torch.concatenate([features['abnormal'], features['normal'], features['prototypes']], axis=0)
    c = torch.concatenate([coords['abnormal'], coords['normal'], coords['prototypes']], axis=0)
    c = (c - c.min(axis=0).values) / (c.max(axis=0).values - c.min(axis=0).values)
    coords['abnormal'] = c[:n_abnormal]
    coords['normal'] = c[n_abnormal:n_abnormal+n_normal]
    coords['prototypes'] = c[n_abnormal+n_normal:]

    dist_feat = 1 - cosine_sim(feat, feat).cpu().numpy()
    dist_feat /= dist_feat.max()
    dist_struct = torch.cdist(c, c, p=2).cpu().numpy()
    dist_struct /= dist_struct.max()
    distances = np.clip(alpha * dist_struct + (1 - alpha) * dist_feat,a_min=0, a_max=None) # clip to avoid negative values due to numerical approximation issues


    n_proj = proj_params['n_images'] if 'n_images' in proj_params.keys() else 1

    for i in range(n_proj):
        if proj_type == 'umap':
            proj_model = umap.UMAP( 
                n_components=2,
                n_neighbors=proj_params['n_neighbors'],
                min_dist=proj_params['min_dist'],
                metric='precomputed'
            )
        elif proj_type == 'tsne':
            proj_model = TSNE(
                n_components=2,
                perplexity=proj_params['perplexity'],
                metric='precomputed',
            )
        embedding = proj_model.fit_transform(distances)

        # plot abnomal points
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            embedding[:features['abnormal'].shape[0], 0], 
            embedding[:features['abnormal'].shape[0], 1],
            c=[(c[0], c[1], 0) for c in coords['abnormal']], 
            # label='Abnormal', 
            alpha=1., 
            marker='x',
            s=20
        )
        # plot normal points
        ax.scatter(
            embedding[n_abnormal:n_abnormal+n_normal, 0], 
            embedding[n_abnormal:n_abnormal+n_normal, 1],
            c=[(c[0], c[1], 0) for c in coords['normal']], 
            # label='Normal', 
            alpha=1., 
            marker='o',
            s=20,
        )
        # plot prototypes
        ax.scatter(
            embedding[n_abnormal+n_normal:, 0], 
            embedding[n_abnormal+n_normal:, 1],
            c=[(c[0], c[1], 0) for c in coords['prototypes']], 
            # label='Prototypes', 
            alpha=1., 
            marker='d',
            s=20,
            edgecolor='black',
            linewidths=0.5
        )

        handles = [
            Line2D([0], [0], label="Abnormal", color='black', marker='x', markersize=4, linestyle=''),
            Line2D([0], [0], label="Normal", color='black', marker='o', markersize=4, linestyle=''),
            Line2D([0], [0], label="Prototypes", color='black', marker='d', markersize=4, linestyle=''),
        ]
                
        ax.set_title(title if title is not None else f"Proj {' '.join(['='.join(map(str,i)) for i in proj_params.items()])} alpha={alpha} / n°{i}")
        ax.tick_params(labelsize=12)

        if save_path is not None:
            if n_proj > 1:
                basename = os.path.splitext(os.path.basename(save_path))[0]
                sp = os.path.join(os.path.dirname(save_path), basename+f'_proj_{i}.png')
            else:
                sp = save_path
            ax.set_title(title if title is not None else f"{proj_type} {' '.join(['='.join(map(str,i)) for i in proj_params.items()])} alpha={alpha}")
            ax.legend(handles=handles, fontsize=14)
            fig.savefig(sp, dpi=400, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

