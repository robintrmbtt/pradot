import torch
from torch import nn
import lightning as pl
import timm
import ot
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import hydra
from torchvision.transforms import Normalize
from omegaconf import DictConfig

from pradot.utils import cosine_sim, flatten_dict_for_save_hp

class ProtoEncoder(nn.Module):
    def __init__(self, input_size=(224, 224),
                 backbone='resnet50',
                 pretrained=True,
                 freeze_backbone=True,
                 out_indices=(3,),
                 nb_proto=2,
                 **kwargs):
        super().__init__()
        self.input_size = tuple(input_size)
        self.out_indices = [out_indices] if isinstance(out_indices, (int,float)) else out_indices
        self.nb_proto = [nb_proto for _ in range(len(self.out_indices))] if isinstance(nb_proto, int) or isinstance(nb_proto, float) else nb_proto

        # set feature extractor
        if pretrained:
            self.input_normalization = Normalize(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))
        else:
            self.input_normalization = nn.Identity()
        
        self.feature_extractor = timm.create_model( #TODO: renommer en self.encoder
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.channels = self.feature_extractor.feature_info.channels()
        self.scales = self.feature_extractor.feature_info.reduction()
        self.grid_shapes = [tuple([int(i/s) for i in self.input_size]) for s in self.scales]

        self.norms = nn.ModuleList()
        for channel, scale in zip(self.channels, self.scales):
            self.norms.append(
                nn.Sequential(
                    # nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                    # nn.LayerNorm(
                    # [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                    # elementwise_affine=True,
                    # ),
                    nn.Identity(),
                )
            )
              
        self.initialize_prototypes()
    
    def initialize_prototypes(self):
        prototypes_features = []
        prototypes_coords = []
        grid_coords = []
        for k, grid_s  in enumerate(self.grid_shapes):
            if self.nb_proto[k] > 1:
                assert isinstance(self.nb_proto[k], int), "nb_proto must be an int when using multiple prototypes"
                coords_proto_x = torch.linspace(0, 1, grid_s[0])
                coords_proto_y = torch.linspace(0, 1, grid_s[1])
                coords_proto = torch.stack(torch.meshgrid(coords_proto_x, coords_proto_y, indexing='ij')).permute(1, 2, 0).reshape(-1, 2)
                coords_proto = coords_proto.repeat(self.nb_proto[k], 1).reshape(-1, 2) # PHW,2
            else:
                coords_proto_x = torch.linspace(0,1,int(grid_s[0]*min(1,self.nb_proto[k]))+1)
                coords_proto_x = (coords_proto_x[1:] + coords_proto_x[:-1]) / 2
                coords_proto_y = torch.linspace(0,1,int(grid_s[1]*min(1,self.nb_proto[k]))+1)
                coords_proto_y = (coords_proto_y[1:] + coords_proto_y[:-1]) / 2
                coords_proto = torch.stack(torch.meshgrid(coords_proto_x, coords_proto_y, indexing='ij')).permute(1, 2, 0).reshape(-1, 2)
            prototypes_coords.append(nn.Parameter(coords_proto, requires_grad=False))

            prototypes_features.append(nn.Parameter(torch.rand(coords_proto.shape[0], self.channels[k]), requires_grad=True)) # PHW,C

            coords_grid_x = torch.linspace(0, 1, grid_s[0])
            coords_grid_y = torch.linspace(0, 1, grid_s[1])
            coords_proto = torch.stack(torch.meshgrid(coords_grid_x, coords_grid_y, indexing='ij')).permute(1, 2, 0).reshape(-1, 2)
            grid_coords.append(nn.Parameter(coords_proto, requires_grad=False))

        self.prototypes_features = nn.ParameterList(prototypes_features)
        self.prototypes_coords = nn.ParameterList(prototypes_coords)
        self.grid_coords = nn.ParameterList(grid_coords)

    def forward(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        features = self.feature_extractor(self.input_normalization(input_tensor))
        return [self.norms[k](feat) for k, feat in enumerate(features)]
                

class PRADOT_IMG(pl.LightningModule):
    def __init__(self, 
                 encoder=None, # DictConfig containing encoder parameters
                 data_params=DictConfig({'img_size': (224, 224)}),
                 optim_params=None,
                 alpha=0.5,
                 fixed_structure=True,
                 ema=0.95,
                 eps=0.01,
                 ot_max_iter=100,
                 log_hp=True,
                 **kwargs):
        super().__init__()
        if log_hp:
            self.save_hyperparameters(flatten_dict_for_save_hp(locals().copy()))
        self.model = hydra.utils.instantiate(encoder, input_size=data_params['img_size'])

        self.data_params = data_params
        self.optim_params = optim_params
        self.alpha = alpha
        self.fixed_structure = fixed_structure
        self.ema = ema
        self.eps = eps
        self.ot_max_iter = ot_max_iter

    def _get_cnn_features(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        return self.model(input_tensor)
    
    def forward(self, x):
        features = self._get_cnn_features(x)
        return features

    def training_step(self, data):
        self.model.feature_extractor.eval()

        x = data['image'].float()

        features = self._get_cnn_features(x)

        bary_feat, bary_struct, ot_plans, ot_losses = self.get_barycenters(features)

        loss = torch.mean(torch.cat(ot_losses))
        loss.requires_grad = True # to fix issue with lightning 
        self.log('train/loss', loss, prog_bar=True, logger=True)

        return {'loss': loss, 'bary_feat': bary_feat, 'bary_struct': bary_struct}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update prototypes
        with torch.no_grad():
            for k in range(len(self.model.prototypes_features)):
                # update prototypes features
                self.model.prototypes_features[k] = (1 - self.ema) * outputs['bary_feat'][k] + self.ema * self.model.prototypes_features[k]
                # self.prototypes_features[k] = nn.functional.normalize(self.barycenters[k], p=2, dim=1)
                self.model.prototypes_features[k].requires_grad = True

                # update prototypes structure
                if not self.fixed_structure:
                    if self.model.nb_proto[k] > 1:
                        struct_update = outputs['bary_struct'][k].reshape(self.model.nb_proto[k], -1, 2).mean(0)
                        struct_update = struct_update.repeat(self.model.nb_proto[k], 1)
                    else:
                        struct_update = outputs['bary_struct'][k]
                    self.model.prototypes_coords[k] = (1 - self.ema) * struct_update + self.ema * self.model.prototypes_coords[k]
    
    def validation_step(self, data):
        self.model.feature_extractor.eval()

        x = data['image'].float()

        features = self._get_cnn_features(x)

        _, _, ot_plans, ot_losses = self.get_barycenters(features)

        loss = torch.mean(torch.cat(ot_losses))
        self.log('val/loss', loss, prog_bar=True, logger=True)

    def test_step(self, item):
        # item must be a single element
        return self.get_anomap(item['image'].float())
        
    def configure_optimizers(self):
        return None

    @torch.no_grad()
    def get_barycenters(self, features):
        feat_barycenters = []
        struct_barycenters = []
        ot_plans = []
        ot_losses = []
        for k in range(len(features)):
            batch_pos = self.model.grid_coords[k].repeat(len(features[k]), 1) # PHW,2
            dist_struct = torch.cdist(batch_pos[None], self.model.prototypes_coords[k][None], p=2)[0] # BHW x PHW
            dist_struct /= dist_struct.max()

            batch_feat = features[k]
            dist_feat = 1 - cosine_sim(batch_feat, self.model.prototypes_features[k]) # BHW x PHW
            dist_feat = dist_feat.reshape(-1, dist_feat.shape[-1])
            dist_feat /= dist_feat.max()

            M = (1-self.alpha) * dist_feat + self.alpha * dist_struct
        
            ot_plan = ot.bregman.sinkhorn(
                a=(torch.ones(M.shape[0]) / M.shape[0]).to(M.device),
                b=(torch.ones(M.shape[1]) / M.shape[1]).to(M.device),
                M=M,
                reg=self.eps,
                method='sinkhorn_log',
                numItermax=self.ot_max_iter,
            ) # BHW x PHW
            ot_losses.append((ot_plan * M).sum().view(1))

            feat = batch_feat.permute(0, 2, 3, 1).reshape(-1, batch_feat.shape[1])
            bary_batch = (ot_plan.T * M.shape[1]) @ feat # PHWxC

            ## get batch barycenter positions
            bary_batch_pos = (ot_plan.T * M.shape[1]) @ batch_pos
        
            feat_barycenters.append(bary_batch)
            struct_barycenters.append(bary_batch_pos)
            ot_plans.append(ot_plan)
    
        return feat_barycenters, struct_barycenters, ot_plans, ot_losses

    @torch.no_grad()
    def get_anomap(self, x, mode='distM'):
        assert mode in ['cosim', 'distM'], "Mode must be one of ['cosim', 'distM']"

        # x is one image of shape (1,3,h,w)
        anomaps = []
        ot_dists = []
        ot_plans = []

        self.eval()
        features = self._get_cnn_features(x)

        for k in range(len(features)):
            batch_feat = features[k]
            b,c,h,w = batch_feat.shape

            batch_pos = self.model.grid_coords[k].repeat(len(features[k]), 1) # PHW,2
            dist_struct = torch.cdist(batch_pos[None], self.model.prototypes_coords[k][None], p=2)[0] # BHW x PHW
            dist_struct /= dist_struct.max()

            dist_feat = 1 - cosine_sim(batch_feat, self.model.prototypes_features[k]) # BHW x PHW
            dist_feat = dist_feat.reshape(-1, dist_feat.shape[-1])
            dist_feat /= dist_feat.max()

            M = (1-self.alpha) * dist_feat + self.alpha * dist_struct

            # ## Version 1 (not used): convert 3D position to distance matrix and compute srFGW
            # if mode == "srfgw":
            #     batch_struct = torch.cdist(batch_pos[None], batch_pos[None], p=2)[0] # BHW x BHW
            #     batch_struct /= batch_struct.max()
            #     proto_struct = torch.cdist(self.model.prototypes_coords[k][None], self.model.prototypes_coords[k][None], p=2)[0] # PHW x PHW
            #     proto_struct /= proto_struct.max()
            #     ot_plan, fgw_logs = ot.gromov.semirelaxed_fused_gromov_wasserstein(
            #         dist_feat,
            #         batch_struct,
            #         proto_struct,
            #         alpha=self.alpha,
            #         max_iter=1000,
            #         log=True,
            #         versbose=False
            #     )  # h*w, p*h*w  
            #     ot_dists.append(fgw_logs['srfgw_dist'])
            
            ## Version 2 : assignment is just the closer for OT distance
            if mode == "distM":
                ot_plan = torch.argmin(M, dim=1)
                ot_plan = torch.zeros_like(M).scatter_(1,ot_plan.unsqueeze(1),1.)

                ot_dists.append((M*ot_plan).sum())

                # if self.hard_assign:
                #     ot_plan = torch.argmax(ot_plan, dim=1)
                #     ot_plan = torch.zeros_like(M).scatter_(1,ot_plan.unsqueeze(1),1.) / M.shape[1]               

            ot_plan *= M.shape[1] # normalization across scales
            ot_plans.append(ot_plan.cpu().numpy())
            if mode == "cosim":
                sim = cosine_sim(batch_feat, self.model.prototypes_features[k]).view(h*w, -1)
                sim = torch.einsum('nP,nP->n', ot_plan, sim).view(1, 1, h, w)
                # upscale to image size
                sim =  -torch.nn.functional.interpolate(sim, size=tuple(self.hparams.data_params.input_size), mode='bilinear', align_corners=True)[0, 0]
            elif mode == "distM":
                sim = (ot_plan * M).sum(1).view(1, 1, h, w)
                # upscale to image size
                sim = torch.nn.functional.interpolate(sim, size=tuple(self.model.input_size), mode='bilinear', align_corners=True)[0, 0] # tuple() sinon le type est listconfig
                
            anomaps.append(sim)

        anomaps = torch.stack(anomaps, dim=0).to('cpu').numpy()
        ot_dists = torch.stack(ot_dists, dim=0).to('cpu').numpy()

        return {'anomap': anomaps, 
                'ot_dist': ot_dists, 
                'ot_plan': ot_plans
                }

    def reconstruct_prototypes(self, save_path, dl=None):
        assert dl is not None or min(self.model.nb_proto) >= 1, "Dataloader must be provided and number of prototypes must be >= 1 for all scales"
        os.makedirs(save_path, exist_ok=True)

        self.eval()

        sims = []
        sp_dist = []
        proto_grid_shapes = []
        for k in range(len(self.model.scales)):
            proto_grid_shapes.append((max(self.model.nb_proto[k],1), 
                                        int(min(np.sqrt(self.model.prototypes_coords[k].shape[0]), self.model.grid_shapes[k][0])),
                                        int(min(np.sqrt(self.model.prototypes_coords[k].shape[0]), self.model.grid_shapes[k][1])))
                                        )
            sims.append(np.zeros(proto_grid_shapes[k][0] * proto_grid_shapes[k][1] * proto_grid_shapes[k][2]))
            sp_dist.append(np.zeros((proto_grid_shapes[k][0],) + tuple(self.model.input_size)))

        recons_proto = [np.zeros((proto_grid_shapes[k][0], 3) + tuple(self.model.input_size)) for k in range(len(self.model.nb_proto))]

        for data in tqdm(dl, total=len(dl), desc='Gathering features for reconstruction...'):
            x = data['image'].float().to(self.device)
            assert x.shape[0] == 1, "Batch size of dataloader must equal 1."

            image_features = self._get_cnn_features(x)

            for k in range(len(image_features)):
                feat = image_features[k]

                bary = self.model.prototypes_features[k]

                b, c, h, w = feat.shape

                features_sim = cosine_sim(feat, bary) 
                features_sim = features_sim.view(b*h*w, -1).detach().cpu().numpy()  # b*h*w, p*h*w
                highest_sim = np.argmax(features_sim, axis=0)  # p*h*w

                dist_flag = (features_sim[highest_sim, np.arange(len(highest_sim))] > sims[k])
                sims[k] = (1 - dist_flag) * sims[k] + features_sim[highest_sim, np.arange(len(highest_sim))] * dist_flag
                dist_flag = dist_flag.reshape(proto_grid_shapes[k][0], proto_grid_shapes[k][1], proto_grid_shapes[k][2])  # p,h,w
                patches = x[0].unfold(1, self.model.scales[k], self.model.scales[k]).unfold(2, 
                                                                                            self.model.scales[k],
                                                                                            self.model.scales[k]
                                                                                            ).detach().cpu().numpy()


                features_sim = features_sim.reshape(-1, proto_grid_shapes[k][0], proto_grid_shapes[k][1], proto_grid_shapes[k][2])
                for p in range(proto_grid_shapes[k][0]):
                    for i in range(proto_grid_shapes[k][1]):
                        for j in range(proto_grid_shapes[k][2]):
                            if dist_flag[p, i, j]:
                                recons_proto[k][p, 
                                                :, 
                                                self.model.scales[k] * i:self.model.scales[k] * (i + 1), 
                                                self.model.scales[k] * j:self.model.scales[k] * (j + 1)
                                                ] = patches[:, i, j, :, :]

        os.makedirs(save_path, exist_ok=True)
        self.recons_prototypes = []
        for k in range(len(self.model.scales)):
            self.recons_prototypes.append(recons_proto[k])
            for p in range(proto_grid_shapes[k][0]):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(np.moveaxis(recons_proto[k][p], 0, -1))
                ax.set_title(f'Scale {self.model.out_indices[k]} - Prototype {p}')
                ax.axis('off')
                fig.tight_layout()
                fig.savefig(os.path.join(save_path, f'scale_{self.model.out_indices[k]}_proto_{p}_sim.png'), dpi=400, bbox_inches='tight')
                plt.close(fig)

    def reconstruct_image(self, M, k, x=None, save_path=None, title=None):
        # x is one image of shape (1,3,h,w)
        self.eval()

        if isinstance(M, torch.Tensor):
            M = M.detach().cpu().numpy()
        best_assign = np.argmax(M, axis=1)

        x_recons = np.zeros((3,) + tuple(self.model.input_size))
        best_p = best_assign // (self.model.grid_shapes[k][0] * self.model.grid_shapes[k][1])
        best_assign = best_assign - best_p * (self.model.grid_shapes[k][0] * self.model.grid_shapes[k][1])
        best_h = best_assign // self.model.grid_shapes[k][0]
        best_w = best_assign % self.model.grid_shapes[k][0]
    
        for i in range(self.model.grid_shapes[k][0]):
            for j in range(self.model.grid_shapes[k][1]):
                n = i * self.model.grid_shapes[k][1] + j
                x_recons[:,
                         i*self.model.scales[k]: (i+1)*self.model.scales[k], 
                         j*self.model.scales[k]: (j+1)*self.model.scales[k]
                         ] = self.recons_prototypes[k][best_p[n],
                                                       :, 
                                                       self.model.scales[k] * best_h[n]:self.model.scales[k] * (best_h[n] + 1),
                                                       self.model.scales[k] * best_w[n]:self.model.scales[k] * (best_w[n] + 1)
                                                       ]
            
        fig, axs = plt.subplots(1,1+int(x is not None), figsize=(6*(1+int(x is not None)), 6))
        axs[0].axis('off')
        axs[0].imshow(np.moveaxis(x_recons, (0, 1, 2), (2, 0, 1)))
        if x is not None:
            axs[1].axis('off')
            axs[1].imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
        if save_path is not None:
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.plot()

    def visualize_assignments(self, M, k, img=None, save_path=None, title=None):
        h, w = self.model.grid_shapes[k]
        assert M.shape[0] == h * w, "La taille de M doit correspondre à la taille de la grille de features."

        y_coords_grid, x_coords_grid = self.model.grid_coords[k].T.detach().cpu().numpy()
        y_coords_grid = y_coords_grid * self.model.input_size[0]
        x_coords_grid = x_coords_grid * self.model.input_size[1]

        y_coords_proto, x_coords_proto = self.model.prototypes_coords[k].T.detach().cpu().numpy()
        y_coords_proto = y_coords_proto * self.model.input_size[0]
        x_coords_proto = x_coords_proto * self.model.input_size[1]

        if isinstance(M, torch.Tensor):
            M = M.detach().cpu().numpy()
        best_assign = np.argmax(M, axis=1)
        best_assign = best_assign % (h*w)

        # Tracer les flèches d'assignation
        fig, ax = plt.subplots(figsize=(6, 6))
        if img is not None:
            ax.imshow(img[0].permute(1, 2, 0).detach().cpu().numpy()[::-1,:,:]) # invert y axis to compensate plt.gca.invert_axis
        ax.plot(x_coords_grid, y_coords_grid, 'o', color='green', alpha=0.5, markersize=3)
        if not self.fixed_structure or self.model.nb_proto[k] < 1:
            ax.plot(x_coords_proto, y_coords_proto, 'o', color='red', alpha=0.5, markersize=3)
        for i in range(h * w):
            if x_coords_proto[best_assign[i]] == x_coords_grid[i] and y_coords_proto[best_assign[i]] == y_coords_grid[i]:
                ax.scatter(x_coords_grid[i], y_coords_grid[i], color='blue', alpha=0.5, s=20)
            ax.arrow(
                x_coords_grid[i], y_coords_grid[i],
                x_coords_proto[best_assign[i]] - x_coords_grid[i],
                y_coords_proto[best_assign[i]] - y_coords_grid[i],
                color='blue', alpha=0.5, head_width=3., head_length=3., length_includes_head=True
            )

        ax.invert_yaxis()  # Inverser l'axe y pour correspondre à l'indexation des images
        ax.set_title(title if title is not None else f'Assignment for scale {self.model.out_indices[k]}')
        ax.axis('off')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path), dpi=400)
            plt.close(fig)
            return None
        
        return fig
    

