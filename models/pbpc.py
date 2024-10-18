import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, FOSTERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
import losses as tpc

# 初始化全局变量 USE_CUPY
USE_CUPY = False

# 尝试导入 CuPy，如果成功，则设置 USE_CUPY 为 True
try:
    import cupy as cp

    USE_CUPY = True
    print("CuPy 导入成功，使用 GPU 加速版本。")
except ImportError:
    import numpy as np

    print("CuPy 导入失败，使用 NumPy 版本。")

EPSILON = 1e-8


class PBPC(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)
        self._protos = []
        self._radius = 0
        self._radiuses = []

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        self.save_checkpoint(
            "{}_{}_{}_{}_{}".format(
                self.args["dataset"],
                self.args["model_name"],
                self.args["init_cls"],
                self.args["increment"],
                self.args["batch_size"],
            )
        )

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes * 4)
        self._network_module_ptr = self._network
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=self.args["num_workers"],
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):

        resume = False
        if self._cur_task in self.args["checkpoint"]:
            self._network.load_state_dict(
                torch.load(
                    "./checkpoint/{}_{}_{}_{}_{}_{}.pkl".format(
                        self.args["dataset"],
                        self.args["model_name"],
                        self.args["init_cls"],
                        self.args["increment"],
                        self.args["batch_size"],
                        self._cur_task,
                    )
                )["model_state_dict"]
            )
            resume = True
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        if not resume:
            self._epoch_num = self.args["epochs"]
            optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"]
            )
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()

    def _build_protos(self):

        features = []
        features_old = []

        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset,
                    batch_size=self.args["batch_size"],
                    shuffle=False,
                    num_workers=4,
                )
                if self._cur_task > 0:
                    for _, _inputs, _targets in idx_loader:
                        for rotate in range(4):
                            _inputs = torch.rot90(_inputs, rotate, dims=(2, 3))
                            _inputs = _inputs.to(self._device, non_blocking=True)
                            feature_old = self.old_network_module_ptr.extract_vector(
                                _inputs
                            )
                            features_old.append(feature_old.cpu().numpy())

                for rotate in range(4):
                    vectors, _ = self._extract_vectors(idx_loader, rotate)
                    features.append(vectors)
                    class_mean = np.mean(vectors, axis=0)
                    self._protos.append(class_mean)
                    cov = np.cov(vectors.T)
                    self._radiuses.append(np.trace(cov) / vectors.shape[1])
            self._radius = np.sqrt(np.mean(self._radiuses))

        if self._cur_task > 0:
            # features = np.array(features).reshape(-1, vectors.shape[1])
            features = np.vstack(features)
            features_old = np.vstack(features_old)
            proto_old = np.array(self._protos[: 4 * self._known_classes])

            if USE_CUPY:
                gap = self._protoReplace_cupy_batch(
                    features_old, features, proto_old, 0.2
                )
                cp.cuda.Stream.null.synchronize()  # 确保所有GPU操作完成
            else:

                gap = self._protoReplace(features_old, features, proto_old, 0.2)

            proto_old += gap
            self._protos[: 4 * self._known_classes] = proto_old.tolist()

    def _protoReplace(self, y1, y2, embedding_old, sigma):
        dy = y2 - y1
        distance = np.sum(
            (
                np.tile(y1[None, :, :], [embedding_old.shape[0], 1, 1])
                - np.tile(embedding_old[:, None, :], [1, y1.shape[0], 1])
            )
            ** 2,
            axis=2,
        )
        w = np.exp(-distance / (2 * sigma**2)) + 0.00001  # +1e-5
        w_norm = w / np.tile(np.sum(w, axis=1)[:, None], [1, w.shape[1]])
        result = np.sum(
            np.tile(w_norm[:, :, None], [1, 1, dy.shape[1]])
            * np.tile(dy[None, :, :], [w.shape[0], 1, 1]),
            axis=1,
        )
        return result

    def _protoReplace_cupy_batch(self, y1, y2, embedding_old, sigma, batch_size=8):

        # # 将输入转换为CuPy数组
        y1 = cp.asarray(y1)
        y2 = cp.asarray(y2)
        embedding_old = cp.asarray(embedding_old)

        dy = y2 - y1
        num_samples = embedding_old.shape[0]
        result = cp.zeros((num_samples, dy.shape[1]))

        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)

            # 当前批次的数据
            embedding_batch = embedding_old[i:batch_end]

            distance = cp.sum(
                (
                    cp.tile(y1[None, :, :], [embedding_batch.shape[0], 1, 1])
                    - cp.tile(embedding_batch[:, None, :], [1, y1.shape[0], 1])
                )
                ** 2,
                axis=2,
            )
            w = cp.exp(-distance / (2 * sigma**2)) + 0.00001  # +1e-5
            w_norm = w / cp.tile(cp.sum(w, axis=1)[:, None], [1, w.shape[1]])
            result[i:batch_end] = cp.sum(
                cp.tile(w_norm[:, :, None], [1, 1, dy.shape[1]])
                * cp.tile(dy[None, :, :], [w.shape[0], 1, 1]),
                axis=1,
            )

        return cp.asnumpy(result)

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_clf, losses_fkd, losses_proto, losses_tpc = 0.0, 0.0, 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc="Training"
            ):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True
                ), targets.to(self._device, non_blocking=True)
                inputs = torch.stack(
                    [torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1
                )
                inputs = inputs.view(-1, 3, inputs.shape[-2], inputs.shape[-1])
                targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                logits, loss_clf, loss_fkd, loss_proto, loss_tpc = (
                    self._compute_pbpc_loss(inputs, targets)
                )
                loss = loss_clf + loss_fkd + loss_proto + loss_tpc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                losses_tpc += loss_tpc.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            print("proto aug number: ",
                int(self._known_classes / (self._total_classes - self._known_classes))
            )
            
            if epoch % 5 != 0:
                info = "Task {}, [{}], Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_tpc {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    self.args["dataset"],
                    epoch + 1,
                    self._epoch_num,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fkd / len(train_loader),
                    losses_proto / len(train_loader),
                    losses_tpc / len(train_loader),
                    train_acc,
                )
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, [{}], Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_tpc {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    self.args["dataset"],
                    epoch + 1,
                    self._epoch_num,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_fkd / len(train_loader),
                    losses_proto / len(train_loader),
                    losses_tpc / len(train_loader),
                    train_acc,
                    test_acc,
                )
            prog_bar.set_description(info)
            logging.info(info)

    def _compute_pbpc_loss(self, inputs, targets):
        
        # features = self._network_module_ptr.extract_vector(inputs)
        features = self._network.extract_vector(inputs)
        logits = self._network.fc(features)["logits"]
        # logits = self._network(inputs)["logits"]
        loss_clf = F.cross_entropy(logits / self.args["temp"], targets)
        
        criterion = tpc.Proxy_Anchor(
            nb_classes=self.args["init_cls"] * 4
            + self.args["increment"] * 4 * self._cur_task,
            sz_embed=512,
            mrg=0.1,
            alpha=32,
        ).to(self._device)

        loss_tpc = criterion(features, targets.squeeze().to(self._device))

        loss_tpc = self.args["lambda_tpc"] * loss_tpc

        if self._cur_task == 0:

            return logits, loss_clf, torch.tensor(0.0), torch.tensor(0.0), loss_tpc

        # features = self._network_module_ptr.extract_vector(inputs)
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_fkd = self.args["lambda_fkd"] * torch.dist(features, features_old, 2)

        # index = np.random.choice(range(self._known_classes),size=self.args["batch_size"],replace=True)
        index = np.random.choice(
            range(self._known_classes * 4),
            size=self.args["batch_size"]
            * int(
                self._known_classes / (self._total_classes - self._known_classes)
            ),
            replace=True,
        )
        # print(index)
        # print(np.concatenate(self._protos))
        proto_features = np.array(self._protos)[index]
        # print(proto_features)
        proto_targets = index
        proto_features = (
            proto_features + np.random.normal(0, 1, proto_features.shape) * self._radius
        )
        proto_features = (
            torch.from_numpy(proto_features).float().to(self._device, non_blocking=True)
        )
        proto_targets = torch.from_numpy(proto_targets).to(
            self._device, non_blocking=True
        )

        proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(
            proto_logits / self.args["temp"], proto_targets
        )
        return logits, loss_clf, loss_fkd, loss_proto, loss_tpc

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):

            inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
            inputs = inputs.view(-1, 3, inputs.shape[-2], inputs.shape[-1])
            inputs = inputs.to(self._device)

            with torch.no_grad():
                logits = model(inputs)["logits"]
                outputs = logits[::4, ::4]

                for output in range(1, 4):
                    outputs += logits[output::4, output::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):

            inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
            inputs = inputs.view(-1, 3, inputs.shape[-2], inputs.shape[-1])
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = self._network(inputs)["logits"]
                outputs = logits[::4, ::4]
                for output in range(1, 4):
                    outputs += logits[output::4, output::4]

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        elif hasattr(self, "_protos"):
            y_pred, y_true = self._eval_nme(
                self.test_loader,
                self._protos / np.linalg.norm(self._protos, axis=1)[:, None],
            )
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy
