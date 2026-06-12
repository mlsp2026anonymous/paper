import copy

import ot
import torch
import torch.nn as nn
import numpy as np
import itertools
import skimage.filters as sfil

from sklearn.cluster import KMeans
from torch.nn import BCELoss
from tqdm import tqdm

from models.models import classifier,CLS, ClassMemoryQueue
from models.loss import ConditionalEntropyLoss, Entropy
from utils import adaptive_filling, ubot_CCD, sinkhorn, ubot_CCD2, adaptive_filling2
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F



def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name in globals():
        return globals()[algorithm_name]
    # Methods implemented in isolated modules to keep this file manageable
    _ISOLATED = {
        "RAINCOAT": ("algorithms.raincoat", "RAINCOAT"),
    }
    if algorithm_name in _ISOLATED:
        import importlib
        mod_path, cls_name = _ISOLATED[algorithm_name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.is_uniDA = False
        self.uniDA = True

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()),
            lr=0.005,
        )


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model
    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
    def get_latent_features(self, dataloader):
        feature_set = []
        pred_set = []
        label_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, _) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.feature_extractor(data)
                pred = F.softmax(self.classifier(feature))
                pred_set.append(pred.cpu())
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            pred_set = torch.cat(pred_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set, pred_set

    def evaluate(self, test_loader, trg_private_class, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                if self.uniDA:
                    if src and self.is_uniDA:
                        corr_preds = self.correct_predictions(predictions)
                        loss = F.cross_entropy(corr_preds, labels)
                        #loss = F.cross_entropy(predictions[m], labels[m])
                    else:
                        #m = torch.isin(labels, self.trg_private_class.view((-1)).long().to(self.device), invert=True)
                        m = torch.isin(labels.cpu(), trg_private_class, invert=True)
                        loss = F.cross_entropy(predictions[m], labels[m])
                else:
                    loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.detach().cpu().item())
                #predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels
    def correct_predictions(self, preds):
        return preds

    def decision_function(self, preds):
        confidence, pred = preds.max(dim=1)
        return pred
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError

class UniJDOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        print(configs)
        # device
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        #self.classifier = CLS(configs, temp=hparams['temp']).to(self.device)
        self.classifier = CLS(configs).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        print("Batch Size : ", hparams['batch_size'])
        self.nb_classes = configs.num_classes

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.is_uniDA = True
        self.src_latent_cluster = None
        self.register_buffer("final_threshold", torch.tensor(0.0))

        #self.final_threshold = None
        self.final_threshold = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.threshold_method = self.get_thresholding_method() #sfil.threshold_yen #sfil.threshold_triangle #sfil.threshold_yen
        #self.threshold_method = self.v
        #self.threshold_method = sfil.threshold_yen  # sfil.threshold_triangle #sfil.threshold_yen
        if configs.isFNO:
            feat_dim = configs.final_out_channels + 2*configs.fourier_modes
        else:
            feat_dim = configs.final_out_channels #+ 2 * configs.fourier_modes
        self.memqueue_feat = ClassMemoryQueue(feat_dim, self.nb_classes, hparams['n_batch']).cuda()
        #self.memqueue_preds = MemoryQueue(configs.num_classes, hparams['batch_size'], hparams['n_batch']).cuda()

    '''def v(self, x):
        return 0.5 * sfil.threshold_yen(x) + 0.5 * sfil.threshold_otsu(x)'''
    def init_queue(self, dataloader):
        cnt_i = 0
        for x,y, id in dataloader:
            x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
            feature_ex_s = self.feature_extractor(x)
            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            self.memqueue_feat.update_queue(F.normalize(before_lincls_feat_s), y)
            #self.memqueue_preds.update_queue(y_src, id.clone())
            cnt_i += 1
            if self.memqueue_feat.is_memory_full().all():
                break
        print('Memory after init : ', self.memqueue_feat.is_memory_full())

    def infomax_loss(self, cluster_assignments, eps=1e-10):
        """
        InfoMax loss function to maximize mutual information between inputs and cluster assignments.

        Args:
            cluster_assignments (torch.Tensor): Tensor of shape (N, K) representing the probability distribution
                                                over K clusters for each sample.
            eps (float): Small constant to avoid numerical issues with log.

        Returns:
            torch.Tensor: Scalar loss value (InfoMax loss).
        """
        # Batch size and number of clusters
        N, K = cluster_assignments.shape

        # Step 1: Compute the marginal distribution p(z) over clusters (averaging over samples)
        marginal_prob = cluster_assignments.mean(dim=0)  # Shape (K,)

        # Step 2: Compute H(Z) - Entropy of the marginal distribution (cluster assignments)
        H_Z = -torch.sum(marginal_prob * torch.log(marginal_prob + eps))

        # Step 3: Compute H(Z|X) - Conditional entropy of cluster assignment given input
        H_Z_given_X = -torch.sum(cluster_assignments * torch.log(cluster_assignments + eps)) / N

        # Step 4: InfoMax loss is H(Z) - H(Z|X)
        infomax_loss_value = H_Z - H_Z_given_X

        return infomax_loss_value

    def get_thresholding_method(self):
        return getattr(sfil, self.hparams['threshold_method'])

    '''def threshold_method(self, x):
        if sfil.threshold_yen(x) < 1-1/self.nb_classes:
            return sfil.threshold_yen(x[x>1-1/self.nb_classes])
        return max(sfil.threshold_yen(x), 1-1/self.nb_classes)'''

    def class_centroids(self, x, y):
        # Get the number of classes

        # Reduce the sum of each feature across samples within a class
        class_sums = torch.einsum("ji,jk->ki", x, y.float().cuda())
        # return class_sums
        # Count the number of samples in each class (sum along the sample dimension)
        class_counts = torch.sum(y, dim=0)

        # Avoid division by zero for empty classes
        class_counts[class_counts == 0] = 1

        # Divide class sums by class counts to get centroids
        centroids = (class_sums.T / class_counts).T

        return centroids

    def ini_centroids(self, src_dl):
        ctr_list = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for i, (im_s, label_source, id_s) in enumerate(tqdm(src_dl, desc='testing')):
                im_s = im_s.cuda()
                label_source = label_source.cuda()
                feature_ex_s = self.feature_extractor.forward(im_s)
                before_lincls_feat_s, after_lincls_t = self.classifier(feature_ex_s)
                norm_feat_s = F.normalize(before_lincls_feat_s)
                y_src = torch.eye(self.nb_classes, dtype=torch.int8).cuda()[label_source]
                ctr_list.append(self.class_centroids(norm_feat_s, y_src).cpu())
        self.feature_extractor.train()
        self.classifier.train()
        return torch.stack(ctr_list).mean(dim=0)

    def centroids_target(self, trg_dl, K):
        X = []
        self.feature_extractor.eval()
        with torch.no_grad():
            for i, (im_t, label_target, id_t) in enumerate(tqdm(trg_dl, desc='testing')):
                im_t = im_t.cuda()
                feature_ex_t = self.feature_extractor.forward(im_t)
                before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                X.append(norm_feat_t)
        self.feature_extractor.train()
        X = torch.cat(X, 0).cpu()

        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
        return kmeans.cluster_centers_

    def update_centroids_target(self, X, cen):
        X = X.cpu()
        cen = cen.cpu()
        dd = torch.cdist(X, cen)
        pred_cen = dd.argmin(axis=1)

        for i in pred_cen:
            ix = i == pred_cen
            cenc = X[ix].mean(axis=0)
            cen[i] = 0.9 * cen[i] + 0.1 * cenc

        return cen

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self.src_loader = src_loader
        self.init_queue(src_loader)
        print("Memory State : ", self.memqueue_feat.is_memory_full())
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        self.src_latent_cluster = self.ini_centroids(src_loader).cuda()
        self.trg_latent_cluster = self.centroids_target(trg_loader, self.hparams['K'])
        self.trg_latent_cluster = torch.from_numpy(self.trg_latent_cluster).to(torch.float)

        nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        with torch.no_grad():
            self.network.eval()
            X = self.src_loader.dataset.x_data.cuda()
            Y = self.src_loader.dataset.y_data.numpy()
            _, logits = self.network(X)
            preds = logits.detach().cpu().argmax(axis=1).numpy()
            print("SRC Accuracy : ", (Y == preds).sum() / len(Y))
        self.network.train()
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        cnt_i = 0
        self.trg_feats_mem = []
        self.trg_preds_mem = []
        trg_mem_size = self.hparams["trg_mem_size"]
        with torch.no_grad():
            self.network.eval()
            for x, y, id in trg_loader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feature_ex_t = self.feature_extractor(x)
                before_lincls_feat_t, after_lincls_s = self.classifier(feature_ex_t)
                norm_feat_t = F.normalize(before_lincls_feat_t)
                self.trg_feats_mem.append(norm_feat_t)
                self.trg_preds_mem.append(after_lincls_s)
                # self.memqueue_preds.update_queue(y_src, id.clone())
                cnt_i += after_lincls_s.shape[0]
                if cnt_i > trg_mem_size:
                    break
        self.trg_feats_mem = torch.concatenate(self.trg_feats_mem)[:trg_mem_size]
        self.trg_preds_mem = torch.concatenate(self.trg_preds_mem)[:trg_mem_size]

        if self.hparams['joint_decision']:
            dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
            soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        else:
            soft_trg_tr = F.softmax(self.trg_preds_mem, dim=1)
        conf, preds = soft_trg_tr.max(dim=1)
        #self.final_threshold = self.threshold_method(conf.detach().cpu().numpy())
        new_value = self.threshold_method(conf.detach().cpu().numpy())
        self.final_threshold.data.fill_(new_value)
        #self.final_threshold.data = torch.tensor(new_value, device=self.final_threshold.device)

        #self.register_buffer("final_threshold", torch.tensor(self.final_threshold))
        #self.final_threshold = torch.tensor(self.final_threshold, device=self.final_threshold.device)

        '''X = self.src_loader.dataset.x_data.cuda()
        Y = self.src_loader.dataset.y_data.numpy()
        _, logits = self.network(X)
        preds = logits.detach().cpu().argmax(axis=1).numpy()
        print("SRC Accuracy : ", (Y == preds).sum() / len(Y))'''
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)
            #info_loss = self.infomax_loss(F.softmax(src_pred))

            loss = src_cls_loss #+ info_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def compute_cluster_distance(self, x_test):
        #x = self.src_loader.dataset.x_data
        #y = self.src_loader.dataset.y_data
        x = self.memqueue_feat.mem_feat
        '''y = self.memqueue_feat.mem_id
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()'''

        #x = F.normalize(before_lincls_feat_t)
        x, x_test = x.squeeze(), x_test.squeeze()
        '''nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        print(y.unique(return_counts=True))
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist'''
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(0)
        res = self.memqueue_feat.compute_distances(x_test)
        #print(res)

        # res = F.tanh(res)
        # res = res/res.max()
        d = -1 * res
        # d = 1-res
        d = F.softmax(d, dim=1)

        return d

    def compute_cluster_distance2(self, x_test):
        x = self.src_loader.dataset.x_data
        y = self.src_loader.dataset.y_data
        x, x_test, y = torch.Tensor(x).cuda(), torch.Tensor(x_test), torch.Tensor(y).cuda().long()
        feature_ex_t = self.feature_extractor(x)
        before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

        x = F.normalize(before_lincls_feat_t)
        # _, x, _ = net.forward_extractor_logits(x)
        x, x_test = x.squeeze(), x_test.squeeze()
        nb_classes = self.nb_classes
        res = torch.empty(x_test.shape[0], nb_classes)
        res = torch.zeros_like(res)
        for i, ll in enumerate(range(nb_classes)):
            dist = torch.cdist(x[y == ll], x_test).min(axis=0).values
            res[:, int(ll)] = dist

        d = -1 * res
        d = F.softmax(d, dim=1)
        #d = 1-res/res.max()
        return d.max(axis=0).values

    def joint_decision(self, preds, distance):
        preds = torch.tensor(preds).cuda()
        distance = distance.cuda()
        return F.softmax(preds*distance, dim=1)#* distance
        #return F.softmax(F.softmax(preds, dim=1) * distance, dim=1)  # * distance
        #return 0.5*F.softmax(preds, dim=1) + 0.5*distance # * distance
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        #temp = self.hparams['temp']
        # soft = nn.Softmax(dim=1)
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features
            feature_ex_s = self.feature_extractor(src_x)
            feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(feature_ex_s)
            before_lincls_feat_t, after_lincls_t = self.classifier(feature_ex_t)

            norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            # =====Source Supervision=====

            y_src = torch.eye(after_lincls_s.shape[-1], dtype=torch.int8).cuda()[src_y]
            self.memqueue_feat.update_queue(norm_feat_s, src_y.cuda())
            #self.memqueue_preds.update_queue(y_src, id_source.cuda())
            #print(y_src.shape)
            # print("y_src shape : ", y_src.shape)

            centr = self.class_centroids(norm_feat_s, y_src).detach().cpu()
            src_latent_cluster_copy = 0.9 * self.src_latent_cluster + 0.1 * centr.cuda()
            # TRG Centroids
            trg_latent_cluster_copy = self.update_centroids_target(norm_feat_t.detach().cpu(), self.trg_latent_cluster).cuda()
            before_lincls_feat_cen, after_lincls_cen = self.classifier(trg_latent_cluster_copy)
            trg_soft_cen = torch.nn.functional.softmax(after_lincls_cen).double()

            if self.hparams['joint_decision']:
                dist = self.compute_cluster_distance(norm_feat_t)
                soft_t = self.joint_decision(after_lincls_t, dist)
            else:
                soft_t = F.softmax(after_lincls_t)
            conf, preds = soft_t.max(dim=1)
            threshold = self.threshold_method(conf.detach().cpu().numpy())
            if step == 1: print('threshold : ', threshold)
            mask = (conf < threshold).cuda()
            # print("Detected ODD : ", mask.sum().item())

            # print("Number of ODD : ", (trg_label >= cls_output_dim).sum().item())
            # C0 = torch.zeros((len(norm_feat_s) + 1, len(norm_feat_t))).cuda()

            C0 = torch.zeros((len(norm_feat_s) + len(trg_latent_cluster_copy), len(norm_feat_t))).cuda()
            C_latent = src_latent_cluster_copy.mean(axis=0).unsqueeze(0)

            # nonOOD SRC
            C0[:len(norm_feat_s), ~mask] = torch.cdist(norm_feat_s, norm_feat_t[~mask])
            # OOD Dummy
            C0[len(norm_feat_s):, mask] = torch.cdist(trg_latent_cluster_copy, norm_feat_t[mask])

            maxc = torch.max(C0).item()  # *self.hparams['psi']
            # ODD SRC
            C0[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C0[len(norm_feat_s):, ~mask] = maxc

            C1 = torch.zeros(C0.shape).cuda()
            C_preds = torch.ones((trg_latent_cluster_copy.shape[0], y_src.shape[-1])) / self.nb_classes
            C_preds = C_preds.cuda()

            # nonOOD SRC
            C1[:len(norm_feat_s), ~mask] = torch.cdist(y_src.float(), F.softmax(after_lincls_t)[~mask])
            # OOD Dummy
            C1[len(norm_feat_s):, mask] = torch.cdist(C_preds, F.softmax(after_lincls_t)[mask])
            maxc = torch.max(C1).item()  # *self.hparams['psi']
            # ODD SRC
            C1[:len(norm_feat_s), mask] = maxc
            # nonOOD Dummy
            C1[len(norm_feat_s):, ~mask] = maxc

            C = (self.hparams['alpha'] * C0 + self.hparams['lamb'] * C1)

            with torch.no_grad():

                a, b = ot.unif(C.size(0)), ot.unif(C.size(1))
                ratio = (mask.sum() / len(mask)).detach().cpu().item()
                a[:len(norm_feat_s)] = 0.5 / len(a[:len(norm_feat_s)])  #
                # a[:len(norm_feat_s)] = (1-ratio)/len(a[:len(norm_feat_s)])
                a[len(norm_feat_s):] = 0.5 / len(a[len(norm_feat_s):])  #
                # a[len(norm_feat_s):] = ratio/len(a[len(norm_feat_s):])
                # print(a.sum(), b.sum())

                gamma = ot.unbalanced.mm_unbalanced(a, b, C.detach().cpu().numpy(), reg_m=0.5)
                # gamma = ot.partial.partial_wasserstein(a, b, C.detach().cpu().numpy(), m=temp)
                # gamma = ot.sinkhorn(a, b, C.detach().cpu().numpy(), reg=0.01)
                # gamma = ot.emd(a, b, C.detach().cpu().numpy())
                # mass.append(gamma.sum())
                if step == 1:print('Mass : ', gamma.sum())
                gamma = torch.tensor(gamma).cuda()

            assert not torch.isnan(gamma).any()
            assert not torch.isnan(feature_ex_t).any()
            assert not torch.isnan(feature_ex_s).any()
            assert not torch.isnan(after_lincls_t).any()
            assert not torch.isnan(after_lincls_s).any()

            label_align_loss = (C * gamma)  # .sum()

            label_align_loss_nonOOD = label_align_loss[:len(norm_feat_s), ~mask].sum()
            label_align_loss_OOD = label_align_loss[len(norm_feat_s):, mask].sum()

            #label_align_loss = self.hparams['lamb'] * (label_align_loss_nonOOD + label_align_loss_OOD) / 2
            label_align_loss = (label_align_loss_nonOOD + label_align_loss_OOD) / 2

            criterion = nn.CrossEntropyLoss().cuda()
            loss_cls = self.hparams['src_weight'] * criterion(after_lincls_s, src_y)

            loss_all = loss_cls + label_align_loss #+ info_loss

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_align': label_align_loss.item()}#, 'info_loss': info_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, trg_private_class, src=False):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, preds_list, labels_list = [], [], []



        '''dist_trg_tr = self.compute_cluster_distance(self.trg_feats_mem)
        soft_trg_tr = self.joint_decision(self.trg_preds_mem, dist_trg_tr)
        conf, preds = soft_trg_tr.max(dim=1)
        threshold = self.threshold_method(conf.detach().cpu().numpy())'''

        with torch.no_grad():
            for data, labels, ids in test_loader:
                data = data.float().to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                before_lincls_feat_t, predictions = self.classifier(features)
                soft = F.softmax(predictions)

                if self.hparams['joint_decision']:
                    norm_feat_t = F.normalize(before_lincls_feat_t)
                    dist = self.compute_cluster_distance(norm_feat_t)
                    soft = self.joint_decision(predictions, dist)
                else:
                    soft = F.softmax(predictions)

                # preds_t = soft_t.argmax(dim=1)
                conf, preds = soft.max(dim=1)
                #threshold = self.threshold_method(conf.detach().cpu().numpy())
                #print("Finale Threshold : ", threshold)
                #print(self.final_threshold)
                mask = conf < self.final_threshold #threshold
                if not src:
                    predictions[mask.squeeze()] *= 0

                if self.is_uniDA:
                    mask = labels >= predictions.shape[-1]
                    labels[mask] = predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                # z = torch.zeros((len(predictions), 1))
                # predictions = torch.cat((predictions, z.to(predictions.device)), dim=1)
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        return loss, full_preds, full_labels

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        logits_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, ids) in enumerate(dataloader):
                data = data.to(self.device)
                '''if data.shape[0] == 1:
                    continue'''
                feature = self.feature_extractor(data)
                _, logit = self.classifier(feature)
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
                logits_set.append(logit.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
            logits_set = torch.cat(logits_set, dim=0)
        return feature_set, label_set, logits_set

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

