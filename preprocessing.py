import torch
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional, Any
import scipy.sparse as sp


class BBDefinedError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


def get_normalized_adj(A: np.ndarray) -> np.ndarray:
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 1e-5] = 1e-5
    diag   = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X: np.ndarray,
                     num_timesteps_input: int,
                     num_timesteps_output: int,
                     means: np.ndarray,
                     stds: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    indices  = [(i, i + num_timesteps_input + num_timesteps_output)
                for i in range(X.shape[2] - num_timesteps_input - num_timesteps_output + 1)]
    features, target = [], []
    for i, j in indices:
        features.append(X[:, :, i:i + num_timesteps_input].transpose((0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input:j] * stds[0] + means[0])
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))


class TrafficDataset(Dataset):

    def __init__(self,
                 data_args: Dict,
                 task_args: Dict,
                 stage: str = 'source',
                 test_data: str = 'metr-la',
                 add_target: bool = True,
                 target_days: int = 3,
                 enable_augmentation: bool = False,
                 minibatch_size: int = 4096):
        super(TrafficDataset, self).__init__()

        self.data_args           = data_args
        self.task_args           = task_args
        self.his_num             = task_args['his_num']
        self.pred_num            = task_args['pred_num']
        self.stage               = stage
        self.add_target          = add_target
        self.test_data           = test_data
        self.target_days         = target_days
        self.enable_augmentation = enable_augmentation
        self.minibatch_size      = minibatch_size

        self.A_list                 = {}
        self.edge_index_list        = {}
        self.edge_attr_list         = {}
        self.node_feature_list      = {}
        self.x_list                 = {}
        self.y_list                 = {}
        self.means_list             = {}
        self.stds_list              = {}
        self.spatial_features_list  = {}
        self.temporal_features_list = {}
        self.chaos_features_list    = {}

        self.load_data(stage, test_data)

        if self.add_target and stage == 'source':
            self.data_list = np.append(self.data_list, self.test_data)

    def load_data(self, stage: str, test_data: str):
        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
        elif stage in ['target', 'target_maml', 'test', 'validation']:
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError(f'Unsupported stage: {stage}')

        for dataset_name in self.data_list:
            self._load_dataset(dataset_name, stage)

        if stage == 'source' and self.add_target:
            self._load_dataset(test_data, 'target')

    def _load_dataset(self, dataset_name: str, stage: str):
        dataset_config = self.data_args[dataset_name]
        A              = np.load(dataset_config['adjacency_matrix_path'])
        edge_index, edge_attr, node_feature = self.get_attr_func(dataset_config['adjacency_matrix_path'])

        self.A_list[dataset_name]            = torch.from_numpy(get_normalized_adj(A))
        self.edge_index_list[dataset_name]   = edge_index
        self.edge_attr_list[dataset_name]    = edge_attr
        self.node_feature_list[dataset_name] = node_feature

        X = np.load(dataset_config['dataset_path']).transpose((1, 2, 0)).astype(np.float32)
        means = np.mean(X, axis=(0, 2))
        X     = X - means.reshape(1, -1, 1)
        stds  = np.std(X, axis=(0, 2))
        stds  = np.where(stds == 0, 1, stds)
        X     = X / stds.reshape(1, -1, 1)

        if stage == 'source':
            X_processed = X[:, :, :int(X.shape[2] * 0.7)]
        elif stage in ['target', 'target_maml']:
            X_processed = X[:, :, :288 * self.target_days]
        elif stage == 'validation':
            X_processed = X[:, :, int(X.shape[2] * 0.7):int(X.shape[2] * 0.8)]
        elif stage == 'test':
            X_processed = X[:, :, int(X.shape[2] * 0.8):]
        else:
            X_processed = X

        x_inputs, y_outputs = generate_dataset(X_processed, self.his_num, self.pred_num, means, stds)
        self.x_list[dataset_name]     = x_inputs.float()
        self.y_list[dataset_name]     = y_outputs.float()
        self.means_list[dataset_name] = means
        self.stds_list[dataset_name]  = stds

        self._extract_spatial_temporal_features(dataset_name, A, X_processed)

    def _extract_spatial_temporal_features(self, dataset_name: str, A: np.ndarray, X: np.ndarray):
        num_nodes          = A.shape[0]
        degree_centrality  = np.sum(A, axis=1) / (num_nodes - 1)

        A_sparse   = sp.csr_matrix(A)
        distances  = sp.csgraph.shortest_path(A_sparse, directed=False)
        closeness_centrality = np.zeros(num_nodes)
        for i in range(num_nodes):
            valid_d = distances[i][distances[i] != np.inf]
            if len(valid_d) > 1:
                closeness_centrality[i] = (len(valid_d) - 1) / np.sum(valid_d[1:])

        spatial_features = np.column_stack([degree_centrality, closeness_centrality])

        flow_variance = np.var(X[:, 0, :], axis=1)
        flow_trend    = np.diff(X[:, 0, :], axis=1).mean(axis=1) if X.shape[2] > 1 else np.zeros(num_nodes)
        peak_hours    = np.argmax(X[:, 0, :], axis=1) % 24 if X.shape[2] > 0 else np.zeros(num_nodes)
        temporal_features = np.column_stack([flow_variance, flow_trend, peak_hours])

        chaos_features = self._extract_chaos_features_for_nodes(X, dataset_name)

        self.spatial_features_list[dataset_name]  = torch.from_numpy(spatial_features).float()
        self.temporal_features_list[dataset_name] = torch.from_numpy(temporal_features).float()
        if chaos_features is not None:
            self.chaos_features_list[dataset_name] = torch.from_numpy(chaos_features).float()

    def _extract_chaos_features_for_nodes(self, X: np.ndarray, dataset_name: str) -> Optional[np.ndarray]:
        from model.model import ChaosAnalyzer

        chaos_analyzer = ChaosAnalyzer()
        num_nodes      = X.shape[0]

        max_time_steps = min(X.shape[2], 1000)
        if X.shape[2] > max_time_steps:
            step_size   = X.shape[2] // max_time_steps
            time_indices = np.arange(0, X.shape[2], step_size)[:max_time_steps]
            X_sampled   = X[:, :, time_indices]
        else:
            X_sampled = X

        chaos_features_list = []
        max_nodes = min(num_nodes, 50)

        for node_idx in range(max_nodes):
            time_series = X_sampled[node_idx, 0, :]
            if len(time_series) > 10 and np.std(time_series) > 1e-6:
                feats = chaos_analyzer.extract_chaos_features(time_series)[:10]
            else:
                feats = np.zeros(10)
            chaos_features_list.append(feats)

        while len(chaos_features_list) < num_nodes:
            chaos_features_list.append(np.zeros(10))

        chaos_features_array = np.array(chaos_features_list)
        print(f"Extracted chaos features for {dataset_name}: shape {chaos_features_array.shape}")
        return chaos_features_array

    def get_attr_func(self, matrix_path: str,
                      edge_feature_matrix_path: Optional[str] = None,
                      node_feature_path: Optional[str] = None) -> Tuple:
        matrix = np.load(matrix_path)
        a, b, edge_attr = [], [], []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] > 0:
                    a.append(i)
                    b.append(j)
                    edge_attr.append(matrix[i][j])
        return (torch.tensor([a, b], dtype=torch.long),
                torch.tensor(edge_attr, dtype=torch.float),
                None)

    def create_wave_data(self, x_data: torch.Tensor, y_data: torch.Tensor,
                         select_dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, nodes, time_steps, features = x_data.shape
        support_data = x_data.permute(0, 2, 1, 3).contiguous().view(batch_size, time_steps, nodes * features)

        if (select_dataset in self.spatial_features_list and support_data.size(-1) < 1000):
            spatial_features  = self.spatial_features_list[select_dataset]
            temporal_features = self.temporal_features_list[select_dataset]

            max_spatial  = min(spatial_features.shape[1], 2)
            max_temporal = min(temporal_features.shape[1], 2)
            spatial_sub  = spatial_features[:, :max_spatial]
            temporal_sub = temporal_features[:, :max_temporal]

            features_to_add = [spatial_sub, temporal_sub]
            if (select_dataset in self.chaos_features_list and
                    support_data.size(-1) + spatial_sub.numel() + temporal_sub.numel() < 800):
                chaos_feat = self.chaos_features_list[select_dataset]
                features_to_add.append(chaos_feat[:, :min(chaos_feat.shape[1], 3)])

            total_additional = sum(f.shape[0] * f.shape[1] for f in features_to_add)
            if support_data.size(-1) + total_additional <= 500:
                expanded = [
                    f.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, -1, -1)
                     .reshape(batch_size, time_steps, -1)
                    for f in features_to_add
                ]
                support_data = torch.cat([support_data] + expanded, dim=-1)

        if y_data.dim() == 2:
            y_data = y_data.unsqueeze(-1)

        return support_data, y_data.permute(0, 2, 1).contiguous()

    def apply_augmentation(self, x_data: torch.Tensor, noise_std: float = 0.005) -> torch.Tensor:
        if not self.enable_augmentation:
            return x_data

        if random.random() < 0.2:
            x_data = x_data + torch.randn_like(x_data) * noise_std

        if random.random() < 0.15:
            _, _, time_steps, _ = x_data.shape
            shift = random.randint(-1, 1)
            if shift != 0 and time_steps > abs(shift):
                x_data = (torch.cat([x_data[:, :, shift:], x_data[:, :, :shift]], dim=2) if shift > 0
                          else torch.cat([x_data[:, :, -shift:], x_data[:, :, :-shift]], dim=2))

        if random.random() < 0.1:
            x_data = x_data * (1.0 + (random.random() - 0.5) * 0.1)

        return x_data

    def __getitem__(self, index: int) -> Tuple[Data, torch.Tensor]:
        if self.stage in ('source', 'target_maml'):
            select_dataset = random.choice(self.data_list)
            total_samples  = self.x_list[select_dataset].shape[0]
            sample_size    = min(self.minibatch_size, total_samples)
            indices        = torch.randperm(total_samples)[:sample_size]
            x_data         = self.x_list[select_dataset][indices]
            y_data         = self.y_list[select_dataset][indices]
        else:
            select_dataset = self.data_list[0]
            x_data         = self.x_list[select_dataset][index:index + 1]
            y_data         = self.y_list[select_dataset][index:index + 1]

        x_data   = self.apply_augmentation(x_data)
        node_num = self.A_list[select_dataset].shape[0]
        data_i   = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name  = select_dataset

        if select_dataset in self.spatial_features_list:
            data_i.spatial_features  = self.spatial_features_list[select_dataset]
            data_i.temporal_features = self.temporal_features_list[select_dataset]

        return data_i, self.A_list[select_dataset]

    def __len__(self) -> int:
        if self.stage == 'source':
            # Keep the total data seen per epoch proportional to what the old
            # code intended, but cap at a hardware-appropriate ceiling so
            # epochs finish in a reasonable time:
            #   GPU  → up to 200 steps  (larger minibatch, fast kernel)
            #   CPU  → up to 100 steps  (smaller wall-clock budget)
            # Each step already draws `minibatch_size` samples, so the actual
            # sample volume per epoch is minibatch_size × max_steps.
            max_steps = 200 if torch.cuda.is_available() else 100
            base_steps = 500_000
            base_batch = int(self.task_args.get('batch_size', 8))
            uncapped   = max(1, (base_steps * base_batch) // self.minibatch_size)
            return min(uncapped, max_steps)
        return self.x_list[self.data_list[0]].shape[0]

    def get_meta_task_batch(self, task_num: int,
                            support_size: int = None,
                            query_size: int = None) -> Tuple[List, List, List, List]:
        if support_size is None:
            support_size = self.task_args['batch_size']
        if query_size is None:
            query_size = self.task_args['batch_size']

        spt_task_data, qry_task_data         = [], []
        spt_task_A_wave, qry_task_A_wave     = [], []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(task_num):
            select_dataset = random.choice(self.data_list)
            total_samples  = self.x_list[select_dataset].shape[0]
            all_indices    = torch.randperm(total_samples)

            support_x = self.apply_augmentation(self.x_list[select_dataset][all_indices[:support_size]])
            support_y = self.y_list[select_dataset][all_indices[:support_size]]
            query_x   = self.apply_augmentation(self.x_list[select_dataset][all_indices[support_size:support_size + query_size]])
            query_y   = self.y_list[select_dataset][all_indices[support_size:support_size + query_size]]

            node_num     = self.A_list[select_dataset].shape[0]
            support_data = Data(node_num=node_num, x=support_x, y=support_y)
            support_data.edge_index = self.edge_index_list[select_dataset]
            support_data.data_name  = select_dataset
            query_data   = Data(node_num=node_num, x=query_x,   y=query_y)
            query_data.edge_index   = self.edge_index_list[select_dataset]
            query_data.data_name    = select_dataset

            if select_dataset in self.spatial_features_list:
                support_data.spatial_features  = self.spatial_features_list[select_dataset]
                support_data.temporal_features = self.temporal_features_list[select_dataset]
                query_data.spatial_features    = self.spatial_features_list[select_dataset]
                query_data.temporal_features   = self.temporal_features_list[select_dataset]

            A_wave = self.A_list[select_dataset].float()
            spt_task_data.append(support_data.to(device))
            spt_task_A_wave.append(A_wave.to(device))
            qry_task_data.append(query_data.to(device))
            qry_task_A_wave.append(A_wave.to(device))

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave

    def get_enhanced_wave_batch(self, task_num: int,
                                support_size: int = None,
                                query_size: int = None) -> Tuple[List, List]:
        spt_task_data, _, qry_task_data, _ = self.get_meta_task_batch(task_num, support_size, query_size)

        support_batches, query_batches = [], []
        for spt_data, qry_data in zip(spt_task_data, qry_task_data):
            city_name  = spt_data.data_name
            spt_support, spt_query = self.create_wave_data(spt_data.x, spt_data.y, city_name)
            qry_support, qry_query = self.create_wave_data(qry_data.x, qry_data.y, city_name)

            meta_context = None
            if hasattr(spt_data, 'spatial_features') and hasattr(spt_data, 'temporal_features'):
                meta_context = torch.cat([
                    spt_data.spatial_features.mean(dim=0),
                    spt_data.temporal_features.mean(dim=0)
                ], dim=0).unsqueeze(0)

            support_batches.append((spt_support, spt_query, meta_context, city_name))
            query_batches.append((qry_support, qry_query, meta_context, city_name))

        return support_batches, query_batches

    def get_dataset_info(self, dataset_name: str) -> Dict:
        if dataset_name not in self.data_args:
            raise ValueError(f"Dataset {dataset_name} not found in data_args")
        config = self.data_args[dataset_name]
        return {
            'name':               dataset_name,
            'node_num':           config['node_num'],
            'time_step':          config['time_step'],
            'speed_mean':         config['speed_mean'],
            'speed_std':          config['speed_std'],
            'adjacency_shape':    self.A_list.get(dataset_name, torch.tensor([])).shape,
            'data_samples':       self.x_list[dataset_name].shape[0] if dataset_name in self.x_list else 0,
            'has_spatial_features': dataset_name in self.spatial_features_list,
            'has_temporal_features': dataset_name in self.temporal_features_list,
        }

    def get_all_datasets_info(self) -> Dict:
        return {name: self.get_dataset_info(name) for name in self.data_list}


class DataManager:
    def __init__(self, data_args: Dict, task_args: Dict):
        self.data_args = data_args
        self.task_args = task_args
        self.datasets  = {}

    def create_dataset(self, stage: str, test_data: str, **kwargs) -> TrafficDataset:
        dataset = TrafficDataset(
            data_args=self.data_args, task_args=self.task_args,
            stage=stage, test_data=test_data, **kwargs
        )
        self.datasets[f"{stage}_{test_data}"] = dataset
        return dataset

    def create_all_dataloaders(self, test_data: str, target_days: int = 3,
                               enable_augmentation: bool = False,
                               minibatch_size: int = 4096) -> Dict[str, Any]:
        dataloaders = {}

        source_dataset = self.create_dataset('source', test_data,
                                              enable_augmentation=enable_augmentation,
                                              minibatch_size=minibatch_size)
        dataloaders['source'] = DataLoader(source_dataset, batch_size=1, shuffle=True,
                                           num_workers=0, pin_memory=False, drop_last=True)

        validation_dataset = self.create_dataset('validation', test_data, minibatch_size=minibatch_size)
        val_batch = max(1, min(minibatch_size, len(validation_dataset)))
        dataloaders['validation'] = DataLoader(validation_dataset, batch_size=val_batch,
                                               shuffle=False, num_workers=0, pin_memory=False)

        target_dataset = self.create_dataset('target', test_data, target_days=target_days,
                                              minibatch_size=minibatch_size)
        dataloaders['target'] = DataLoader(target_dataset, batch_size=self.task_args['batch_size'],
                                           shuffle=True, num_workers=0, pin_memory=False)

        test_dataset = self.create_dataset('test', test_data, minibatch_size=minibatch_size)
        test_batch   = max(1, min(minibatch_size, len(test_dataset)))
        dataloaders['test'] = DataLoader(test_dataset, batch_size=test_batch,
                                         shuffle=False, num_workers=0, pin_memory=False)

        return dataloaders

    def get_meta_learning_data(self, test_data: str, support_size: int = 8,
                               query_size: int = 12, minibatch_size: int = 4096) -> TrafficDataset:
        return self.create_dataset('source', test_data, enable_augmentation=True,
                                   minibatch_size=minibatch_size)

    def get_dataset_statistics(self) -> Dict:
        return {key: dataset.get_all_datasets_info() for key, dataset in self.datasets.items()}


class FewShotSampler:
    def __init__(self, dataset: TrafficDataset, support_size: int = 8, query_size: int = 12):
        self.dataset      = dataset
        self.support_size = support_size
        self.query_size   = query_size

    def sample_episode(self) -> Tuple[Data, Data, torch.Tensor]:
        select_dataset = random.choice(self.dataset.data_list)
        total_samples  = self.dataset.x_list[select_dataset].shape[0]
        all_indices    = torch.randperm(total_samples)

        support_x = self.dataset.apply_augmentation(self.dataset.x_list[select_dataset][all_indices[:self.support_size]])
        support_y = self.dataset.y_list[select_dataset][all_indices[:self.support_size]]
        query_x   = self.dataset.apply_augmentation(self.dataset.x_list[select_dataset][all_indices[self.support_size:self.support_size + self.query_size]])
        query_y   = self.dataset.y_list[select_dataset][all_indices[self.support_size:self.support_size + self.query_size]]

        node_num     = self.dataset.A_list[select_dataset].shape[0]
        support_data = Data(node_num=node_num, x=support_x, y=support_y)
        support_data.edge_index = self.dataset.edge_index_list[select_dataset]
        support_data.data_name  = select_dataset
        query_data   = Data(node_num=node_num, x=query_x,   y=query_y)
        query_data.edge_index   = self.dataset.edge_index_list[select_dataset]
        query_data.data_name    = select_dataset

        return support_data, query_data, self.dataset.A_list[select_dataset].float()

    def sample_batch(self, batch_size: int) -> List[Tuple[Data, Data, torch.Tensor]]:
        return [self.sample_episode() for _ in range(batch_size)]


def Traffic_data(*args, **kwargs):
    return TrafficDataset(*args, **kwargs)


def create_enhanced_dataloaders(config: Dict, test_dataset: str, target_days: int = 3) -> Dict:
    data_manager    = DataManager(config['data'], config['task'])
    _default_mb     = 2048 if torch.cuda.is_available() else 4096
    minibatch_size  = int(config.get('training', {}).get('minibatch_size', _default_mb))
    return data_manager.create_all_dataloaders(
        test_data=test_dataset,
        target_days=target_days,
        enable_augmentation=config.get('training', {}).get('enable_augmentation', False),
        minibatch_size=minibatch_size,
    )