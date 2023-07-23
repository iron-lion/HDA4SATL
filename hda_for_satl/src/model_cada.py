import glob
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics.cluster import adjusted_rand_score
from src.utils import split_masked_cells
import src.final_classifier as  classifier
import src.model_block as bmodel


class LINEAR_LOGSOFTMAX(nn.Module):
    """
    Defines the forward pass of the LINEAR_LOGSOFTMAX final classifier.
    """
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


class CAE(nn.Module):
    def __init__(self, params, i_len, o_len, e_dim = [1024], d_dim = [1024], l_dim=128, bn=False, relu=0.2):
        """
        Initializes the model by setting the architecture and hyperparameters.

        Args:
                params (object): Object containing model parameters
                i_len (int): Length of the input channel
                o_len (int): Length of the output channel
                e_dim (list): List of dimensions for the embedding layers
                d_dim (list): List of dimensions for the decoding layers
                l_dim (int): Dimension of the latent space
                bn (bool): Flag indicating whether to use batch normalization
                relu (float): ReLU slope
        """
        super(CAE, self).__init__()

        self.input_channel_len = i_len
        self.output_channel_len = o_len
        self.embeding = e_dim
        if len(d_dim) == 0:
            self.decoding = e_dim[::-1] 
        else:
            self.decoding = d_dim
        self.latent = l_dim
        self.bn = bn
        self.relu = relu

        self.weight_recon = 0.0001
        self.weight_kld = 0.0
        self.weight_dist = 1.0

        self.device = params.device
        self.current_epoch = 0
        self.epochs = params.epochs
        self.lr = params.learning_rate
        self.lrS = params.lr_scheduler_step
        self.lrG = params.lr_scheduler_gamma
        self.batch_size = params.batch_size

        self.optim_init = False
 
    def weight_step(self):
        """
        Updates the weight values used in the loss function based on the current epoch.
        """
        self.weight_kld = min(0.25, max(0, self.current_epoch - 6) * 0.003)
        self.weight_dist = 8.13
        self.weight_recon = min(2.37, max(0, self.current_epoch - 21) * 0.05)


    def init_model(self):
        """
        Initializes the model by creating the feature encoder and decoder layers.
        """
        self.feature_encoder1 = bmodel.VGEPEncoder(self.input_channel_len, self.embeding, self.latent, bn=self.bn, relu=self.relu).to(self.device)
        self.feature_decoder1 = bmodel.ADecoder(self.input_channel_len, self.decoding, self.latent, bn=self.bn, relu=self.relu).to(self.device)
        self.feature_encoder2 = bmodel.VGEPEncoder(self.output_channel_len, self.embeding, self.latent, bn=self.bn, relu=self.relu).to(self.device)
        self.feature_decoder2 = bmodel.ADecoder(self.output_channel_len, self.decoding, self.latent, bn=self.bn, relu=self.relu).to(self.device)

        bmodel.weights_init(self.feature_encoder1)
        bmodel.weights_init(self.feature_decoder1)
        bmodel.weights_init(self.feature_encoder2)
        bmodel.weights_init(self.feature_decoder2)


    def model_train(self):
        """
        Sets the model in training mode for all the encoder and decoder layers.
        """
        self.feature_encoder1.train()
        self.feature_encoder2.train()
        self.feature_decoder1.train()
        self.feature_decoder2.train()


    def model_eval(self):
        """
        Sets the model in evaluation mode for all the encoder and decoder layers.
        """
        self.feature_encoder1.eval()
        self.feature_encoder2.eval()
        self.feature_decoder1.eval()
        self.feature_decoder2.eval()


    def reparameterize(self, noise, mu, logvar):
        """
        Applies the reparameterization trick to the latent variables.

        Args:
            noise (bool): Flag indicating whether to add random noise to the latent variables
            mu (tensor): Mean values of the latent variables
            logvar (tensor): Log-variance values of the latent variables

        Returns:
            tensor: Reparameterized latent variables
        """
        if noise:
            sigma = torch.exp(logvar).to(self.device)
            eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1).to(self.device)
            eps  = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            return mu


    def encode1(self, x):
        """
        Encodes the input data using the feature encoder.

        Args:
            x (tensor): Input data

        Returns:
            tensor: Encoded data
        """
        self.model_eval()
        z = self.feature_encoder1(x)
        return z


    def encode2(self, x):
        """
        Encodes the input data using the feature encoder.

        Args:
            x (tensor): Input data

        Returns:
            tensor: Encoded data
        """
        self.model_eval()
        z = self.feature_encoder2(x)
        return z


    def init_optim(self):
        """
        Initializes the optimizers and learning rate schedulers for the model parameters.

        Returns:
            tuple: Tuple containing the optimizer and scheduler objects for the feature encoder and decoder of both channels
        """
        if self.optim_init is False:
            self.fe_optim1 = torch.optim.Adam(params=self.feature_encoder1.parameters(), lr=self.lr)
            self.fd_optim1 = torch.optim.Adam(params=self.feature_decoder1.parameters(), lr=self.lr)
            self.fe_optim2 = torch.optim.Adam(params=self.feature_encoder2.parameters(), lr=self.lr)
            self.fd_optim2 = torch.optim.Adam(params=self.feature_decoder2.parameters(), lr=self.lr)


            self.feature_encoder_scheduler1 = StepLR(self.fe_optim1,step_size=self.lrS,gamma=self.lrG) # decay LR
            self.feature_decoder_scheduler1 = StepLR(self.fd_optim1,step_size=self.lrS,gamma=self.lrG) # decay LR
            self.feature_encoder_scheduler2 = StepLR(self.fe_optim2,step_size=self.lrS,gamma=self.lrG) # decay LR
            self.feature_decoder_scheduler2 = StepLR(self.fd_optim2,step_size=self.lrS,gamma=self.lrG) # decay LR

            self.optim_init = True
        else:
            pass
        
        return self.fe_optim1, self.fd_optim1,\
                self.feature_encoder_scheduler1, self.feature_decoder_scheduler1, \
                self.fe_optim2, self.fd_optim2,\
                self.feature_encoder_scheduler2, self.feature_decoder_scheduler2


    def recon_loss(self, exp1, exp2, lab1, lab2):
        """
        Calculates the reconstruction loss for the given input samples.

        Args:
            exp1 (array-like): Input samples for the first channel of the CAE
            exp2 (array-like): Input samples for the second channel of the CAE
            lab1 (array-like): Labels for the first channel of the CAE
            lab2 (array-like): Labels for the second channel of the CAE

        Returns:
            tensor: Reconstruction loss
        """
        v_exp1 = Variable(torch.tensor(exp1).float()).to(self.device)
        v_exp2 = Variable(torch.tensor(exp2).float()).to(self.device)

        #mse = nn.MSELoss().to(self.device)
        mse = nn.L1Loss(reduction='sum').to(self.device)

        mu_exp1, logvar_exp1 = self.feature_encoder1(v_exp1)
        mu_exp2, logvar_exp2 = self.feature_encoder2(v_exp2)
        
        z1 = self.reparameterize(True, mu_exp1, logvar_exp1)
        z2 = self.reparameterize(True, mu_exp2, logvar_exp2)

        r1 = self.feature_decoder1(z1)
        r2 = self.feature_decoder2(z2)
        loss_reconst = mse(r1, v_exp1) + mse(r2, v_exp2)

        cr1 = self.feature_decoder1(z2)
        cr2 = self.feature_decoder2(z1)
        #mask = torch.Tensor([1.0 if lab1[i] == lab2[i] else 0.0 for i in range(len(lab1))])
        #cr1 = torch.matmul(mask, cr1)
        #cr2 = torch.matmul(mask, cr2)
        #v1 = torch.matmul(mask, v_exp1)
        #v2 = torch.matmul(mask, v_exp2)
        loss_cross_reconst = mse(cr1, v_exp1) + mse(cr2, v_exp2)


        #label = [1 if lab1[i] == lab2[i] else 0 for i in range(len(lab1)) ]
        distance = torch.sqrt(torch.sum((mu_exp1 - mu_exp2) ** 2, dim=1) + \
                           torch.sum((torch.sqrt(logvar_exp1.exp()) - torch.sqrt(logvar_exp2.exp()))**2, \
                        dim = 1))
        distance = distance.sum()

        #dist = []
        #for i in range(len(label)):
        #    dist.append(label[i] * distance[i])
        #distance = sum(dist)/(sum(label)+1e6)
        """

        label = [0 if lab1[i] == lab2[i] else 1 for i in range(len(lab1)) ]
        dist = torch.nn.functional.pairwise_distance(mu_exp1, mu_exp2)
        distance = []
        for i in range(len(label)):
            dloss = (1 - label[i]) * torch.pow(dist[i], 2) \
                    + (label[i]) * torch.pow(torch.clamp(2.0 - dist[i], min=0.0), 2)
            distance.append(dloss)
        distance = sum(distance)/len(label)

        """

        KLD_exp1 = (0.5 * torch.sum(1 + logvar_exp1 - mu_exp1.pow(2) - logvar_exp1.exp()))
        KLD_exp2 = (0.5 * torch.sum(1 + logvar_exp2 - mu_exp2.pow(2) - logvar_exp2.exp()))
        KLD = KLD_exp1 + KLD_exp2

        self.weight_step()
        loss = loss_reconst - self.weight_kld * KLD
        if loss_cross_reconst > 0:
            loss += self.weight_recon * loss_cross_reconst
        if distance > 0:
            loss += self.weight_dist * distance
        #print(loss_reconst.data, loss_cross_reconst.data, KLD.data, distance.data)
        return loss


    def train_step(self, exp1, exp2, lab1, lab2, nonneg=False):
        """
        Performs a single training step for the CAE model.

        Args:
            exp1 (array-like): Input samples for the first channel of the CAE
            exp2 (array-like): Input samples for the second channel of the CAE
            lab1 (array-like): Labels for the first channel of the CAE
            lab2 (array-like): Labels for the second channel of the CAE
            nonneg (bool): Flag indicating whether to enforce non-negativity constraint

        Returns:
            tensor: Loss value for the training step
        """
        #print("> init training...")
        feature_encoder_optim1, feature_decoder_optim1, feature_encoder_scheduler1, feature_decoder_scheduler1,\
        feature_encoder_optim2, feature_decoder_optim2, feature_encoder_scheduler2, feature_decoder_scheduler2 \
                = self.init_optim()
 
        loss = self.recon_loss(exp1, exp2, lab1, lab2)
        
        # training
        self.feature_encoder1.zero_grad()
        self.feature_decoder1.zero_grad()
        self.feature_encoder2.zero_grad()
        self.feature_decoder2.zero_grad()
         
        loss.backward()
        
        feature_encoder_optim1.step()
        feature_decoder_optim1.step()
        feature_encoder_optim2.step()
        feature_decoder_optim2.step()

        feature_encoder_scheduler1.step()
        feature_decoder_scheduler1.step()
        feature_encoder_scheduler2.step()
        feature_decoder_scheduler2.step()
        
        return loss.detach().cpu().data


    def train_vae(self, data_loader):
        """
        Performs the training of the CAE model using the given data loader.

        Args:
            data_loader (object): Data loader object that provides training samples

        Returns:
            list: List of losses at each training iteration
        """
        self.data_loader = data_loader
        losses = []
        self.model_train()

        print('train for reconstruction')
        for epoch in range(0, self.epochs):
            self.current_epoch = epoch

            i=-1
            for iters in range(0, data_loader.ntrain, self.batch_size):
                i+=1

                target, source, target_label, source_label = data_loader.next_batch(self.batch_size)
                loss = self.train_step(target, source, target_label, source_label)

                if i%10==0:
                    print(f'epoch {epoch} - iter {i}, loss {str(loss)[:5]}')

                if i%10==0 and i>0:
                    losses.append(loss)

        # turn into evaluation mode:
        #for key, value in self.encoder.items():
        #    self.encoder[key].eval()
        #for key, value in self.decoder.items():
        #    self.decoder[key].eval()

        return losses

    
    def train_classifier(self):
        """
        Performs the training of the classifier using the encoded representations.

        Returns:
            tuple: Tuple containing the best accuracy for known classes, best accuracy for unknown classes, and best harmonic mean
        """
        m1, v1 = self.encode1(torch.Tensor(self.data_loader.X_seen).float())
        z_seen = self.reparameterize(True, m1, v1).detach().numpy()

        m2, v2 = self.encode2(torch.Tensor(self.data_loader.X_source).float())
        z_source = self.reparameterize(True, m2, v2).detach().numpy()

        lr_cls = 0.001
        classifier_batch_size = 32
        num_classes = 11

        train_X = [torch.from_numpy(z_source).float(), torch.from_numpy(z_seen).float()]
        train_Y = [torch.from_numpy(self.data_loader.y_source).float(), torch.from_numpy(self.data_loader.y_seen).float()]
        train_X = torch.cat(train_X, dim=0)
        train_Y = torch.cat(train_Y, dim=0)

        X_test_seen, X_test_unseen, y_test_seen, y_test_unseen = split_masked_cells(self.data_loader.X_test, self.data_loader.y_test, masked_cells=self.data_loader.remove_col)
        
        mt1, vt1 = self.encode1(torch.Tensor(X_test_seen).float())
        test_seen_X = self.reparameterize(True, mt1, vt1)#.detach().numpy()
        #test_seen_X = torch.from_numpy(test_seen_X).float()
        test_seen_Y = y_test_seen
        test_seen_Y = torch.from_numpy(test_seen_Y).float()

        mt2, vt2 = self.encode1(torch.Tensor(X_test_unseen).float())
        test_novel_X = self.reparameterize(True, mt2, vt2)#.detach().numpy()
        #test_novel_X = torch.from_numpy(test_novel_X).float()
        test_novel_Y = y_test_unseen
        test_novel_Y = torch.from_numpy(test_novel_Y).float()
        cls_seenclasses = np.array((list(set(self.data_loader.y_source) - set(self.data_loader.remove_col))))
        cls_seenclasses = torch.from_numpy(cls_seenclasses).float()
        cls_novelclasses = np.array(self.data_loader.remove_col)
        cls_novelclasses = torch.from_numpy(cls_novelclasses).float()

        clf = LINEAR_LOGSOFTMAX(50, num_classes)
        cls = classifier.CLASSIFIER(clf, train_X, train_Y,
                                        test_seen_X, test_seen_Y,
                                        test_novel_X, test_novel_Y,
                                        cls_seenclasses, cls_novelclasses,
                                        num_classes, 'cpu', lr_cls, 0.5, 1,
                                        classifier_batch_size,
                                        True)

        best_acc_known, best_acc_unknown, best_h = -1, -1, -1
        for k in range(20):
            acc_known, acc_unknown, h = cls.fit()
            best_acc_known = acc_known if acc_known > best_acc_known else best_acc_known
            best_acc_unknown = acc_unknown if acc_unknown > best_acc_unknown else best_acc_unknown
            best_h = h if h > best_h else best_h

        return best_h, best_acc_known, best_acc_unknown
