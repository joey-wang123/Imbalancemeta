import torch
import torch.nn.functional as F
import higher
import random
from gbml.gbml import GBML
from utils import get_accuracy_ANIL, apply_grad, mix_grad

class MAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        return None

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target):
        train_logit = fmodel(train_input)
        inner_loss = F.cross_entropy(train_logit, train_target)
        diffopt.step(inner_loss)
        return None

    def outer_loop(self, batch,  is_train, memory_train = None):
        grad_list = []
        self.network.zero_grad()
        loss_dataset = []
        acc_dataset = []
        for ind in range(len(batch)):

            loss_log = 0
            acc_log = 0
            train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(batch[ind])
            if train_inputs.size(2) == 1:
                train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
            
            if test_inputs.size(2) == 1:
                test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)

            embed_mean = 0
            for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
                with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=is_train) as (fmodel, diffopt):

                    for step in range(self.args.n_inner):
                        self.inner_loop(fmodel, diffopt, train_input, train_target)


                    enc_state = fmodel.encoder(train_input)
                    enc_state = enc_state.view(enc_state.size(0), -1)
                    enc_state = torch.mean(enc_state, dim = 0)
                    embed_mean += enc_state.detach()
                    test_logit = fmodel(test_input)
                    outer_loss = F.cross_entropy(test_logit, test_target)
                    loss_log += outer_loss.item()/len(test_inputs)

                    with torch.no_grad():
                        task_acc = get_accuracy_ANIL(test_logit, test_target).item()/len(test_inputs)
                        acc_log += task_acc
                    if is_train:
                        outer_loss.backward(retain_graph=True)
                        grad_dict = {}
                        for param, nameparam in zip(fmodel.parameters(time=0), fmodel.named_parameters()):
                            if param.requires_grad :
                                if param.grad is not None:
                                    name = nameparam[0]
                                    grad_dict[name] = param.grad
                        grad_list.append(grad_dict)


            loss_dataset.append(loss_log)
            acc_dataset.append(acc_log)
            embed_mean = embed_mean/len(train_inputs)
        if is_train:
            weight = torch.ones(len(grad_list))
            weight = weight / torch.sum(weight)
            grad_meta = mix_grad(grad_list, weight)
            if self.args.memory_only and memory_train:
                replay_grad, repgrad_list = self.rep_grad(self.args, memory_train, is_train)
                grad_list.extend(repgrad_list)
                weight = torch.ones(len(grad_list))
                weight = weight / torch.sum(weight)
                grad_meta = mix_grad(grad_list, weight)

            grad_log = apply_grad(self.network, grad_meta)
            self.outer_optimizer.step()
            
            return loss_dataset, acc_dataset, grad_log, embed_mean
        else:
            return loss_dataset, acc_dataset

    def rep_grad(self, args, memory_train, is_train):
        grad_list = []
        self.network.zero_grad()
        count = self.args.sample
        num_memory = len(memory_train)
        if num_memory<count:
            selectmemory = memory_train
        else:
            samplelist = random.sample(range(num_memory), count)
            selectmemory = []
            for ind in samplelist:
                selectmemory.append(memory_train[ind])


        for selectm in selectmemory:
            for key in selectm:
                    select = selectm[key]
                    train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(select)
                    if train_inputs.size(2) == 1:
                        train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)

                    for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
                        with higher.innerloop_ctx(self.network, self.inner_optimizer, track_higher_grads=is_train) as (fmodel, diffopt):
                            for step in range(self.args.n_inner):
                                self.inner_loop(fmodel, diffopt, train_input, train_target)
                            test_logit = fmodel(test_input)
                            outer_loss = F.cross_entropy(test_logit, test_target)
                            outer_loss.backward()
                            grad_dict = {}
                            for param, nameparam in zip(fmodel.parameters(time=0), fmodel.named_parameters()):
                                if param.requires_grad :
                                    if param.grad is not None:
                                        name = nameparam[0]
                                        grad_dict[name] = param.grad
                            self.network.zero_grad()
                            grad_list.append(grad_dict)

        weight = torch.ones(len(grad_list))
        weight = weight / torch.sum(weight)
        grad = mix_grad(grad_list, weight)

        return grad, grad_list