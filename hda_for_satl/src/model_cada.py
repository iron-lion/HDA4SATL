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
import matplotlib.pyplot as plt

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

        self.beta = params['beta']
        self.cross_recon = params['cross_recon']
        self.dist = params['dist']

        self.weight_recon = self.cross_recon['weight']
        self.weight_kld = self.beta['weight']
        self.weight_dist = self.dist['weight']

        self.device = params['device']
        self.current_epoch = 0
        self.epochs = params['epochs']
        self.lr = params['learning_rate']
        self.lrS = params['lr_scheduler_step']
        self.lrG = params['lr_scheduler_gamma']
        self.batch_size = params['batch_size']

        self.optim_init = False
        self.log_loss = []
        self.log_acc_known = []
        self.log_acc_unknown = []

    def weight_step(self):
        """
        Updates the weight values used in the loss function based on the current epoch.
        """
        self.weight_kld = min(self.beta['weight'], max(0, (self.current_epoch - self.beta['start'])/(self.beta['end'] - self.beta['start']) * self.beta['weight']))
        self.weight_dist = min(self.dist['weight'], max(0, (self.current_epoch - self.dist['start'])/(self.dist['end'] - self.dist['start']) * self.dist['weight']))
        self.weight_recon = min(self.cross_recon['weight'], max(0, (self.current_epoch - self.cross_recon['start'])/(self.cross_recon['end'] - self.cross_recon['start']) * self.cross_recon['weight']))


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
        loss_cross_reconst = mse(cr1, v_exp1) + mse(cr2, v_exp2)


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

        #label = [1 if lab1[i] == lab2[i] else 0 for i in range(len(lab1)) ]
        distance = torch.sqrt(torch.sum((mu_exp1 - mu_exp2) ** 2, dim=1) + \
                           torch.sum((torch.sqrt(logvar_exp1.exp()) - torch.sqrt(logvar_exp2.exp()))**2, \
                        dim = 1))
        distance = distance.sum()

        #dist = []
        #for i in range(len(label)):
        #    dist.append(label[i] * distance[i])
        #distance = sum(dist)/(sum(label)+1e6)


        self.weight_step()
        loss = loss_reconst - self.weight_kld * KLD
        if loss_cross_reconst > 0:
            loss += self.weight_recon * loss_cross_reconst
        if distance > 0:
            loss += self.weight_dist * distance
        #print(loss.data, loss_reconst.data, loss_cross_reconst.data, KLD.data, distance.data)

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

        # training
        self.feature_encoder1.zero_grad()
        self.feature_decoder1.zero_grad()
        self.feature_encoder2.zero_grad()
        self.feature_decoder2.zero_grad()
 
        loss = self.recon_loss(exp1, exp2, lab1, lab2)
               
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
            epoch_loss = 0
            for iters in range(0, data_loader.batch_length, self.batch_size):
                i+=1

                target, source, target_label, source_label = data_loader.next_batch(self.batch_size)
                loss = self.train_step(target, source, target_label, source_label)

                #if i%10==0:
                    #print(f'epoch {epoch} - iter {i}, loss {str(loss)[:5]}')

                #if i%10==0 and i>0:
                epoch_loss =+ loss
            
            self.log_loss.append(epoch_loss/data_loader.ntrain)
            interval = 20
            if epoch%interval == 0:
                _, acc_known, acc_unknown = self.train_classifier()       
                self.log_acc_known += [acc_known.detach().cpu()]*interval
                self.log_acc_unknown += [acc_unknown.detach().cpu()]*interval
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

        with torch.no_grad():
            ### Prep test set
            X_test_seen, X_test_unseen, y_test_seen, y_test_unseen = split_masked_cells(self.data_loader.X_test, self.data_loader.y_test, masked_cells=self.data_loader.remove_col)
            v_x_test_seen = Variable(torch.tensor(X_test_seen).float()).to(self.device)
            v_x_test_unseen = Variable(torch.tensor(X_test_unseen).float()).to(self.device)         

            mt1, vt1 = self.encode1(v_x_test_seen)
            test_seen_X = self.reparameterize(False, mt1, vt1)
            test_seen_Y = y_test_seen
            test_seen_Y = torch.from_numpy(test_seen_Y).to(self.device)

            mt2, vt2 = self.encode1(v_x_test_unseen)
            test_novel_X = self.reparameterize(False, mt2, vt2)
            test_novel_Y = y_test_unseen
            test_novel_Y = torch.from_numpy(test_novel_Y).to(self.device)

            v_x_seen = Variable(torch.tensor(self.data_loader.X_seen).float()).to(self.device)
            v_x_source = Variable(torch.tensor(self.data_loader.X_source).float()).to(self.device)

            m1, v1 = self.encode1(v_x_seen)
            z_seen = self.reparameterize(True, m1, v1)#.detach().numpy()

            m2, v2 = self.encode2(v_x_source)
            z_source = self.reparameterize(True, m2, v2)#.detach().numpy()

            train_Z = [z_source, z_seen]
            y_source = torch.from_numpy(self.data_loader.y_source).to(self.device)
            y_seen = torch.from_numpy(self.data_loader.y_seen).to(self.device)
            train_L = [y_source, y_seen]

            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z))]# if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L))]# if train_Z[i].size(0) != 0]

            train_X = torch.concat(train_X, dim=0)
            train_Y = torch.concat(train_Y, dim=0)

            #print(set(test_novel_Y))
            #print(set(train_Y))
            cls_seenclasses = np.array((list(set(self.data_loader.y_source) - set(self.data_loader.remove_col))))
            cls_seenclasses = torch.from_numpy(cls_seenclasses).to(self.device)
            cls_novelclasses = np.array(self.data_loader.remove_col)
            cls_novelclasses = torch.from_numpy(cls_novelclasses).to(self.device)

        lr_cls = 0.01
        classifier_batch_size = 32
        num_classes = 11
        clf = LINEAR_LOGSOFTMAX(11, num_classes).to(self.device) # latent_size

        cls = classifier.CLASSIFIER(clf, train_X, train_Y,
                                        test_seen_X, test_seen_Y,
                                        test_novel_X, test_novel_Y,
                                        cls_seenclasses, cls_novelclasses,
                                        num_classes, self.device, lr_cls, 0.5, 1,
                                        classifier_batch_size,
                                        True)
        #print(cls.H)
        best_acc_known, best_acc_unknown, best_h = -1, -1, -1
        for k in range(20):
            acc_known, acc_unknown, h = cls.fit()
            best_acc_known = acc_known if acc_known > best_acc_known else best_acc_known
            best_acc_unknown = acc_unknown if acc_unknown > best_acc_unknown else best_acc_unknown
            best_h = h if h > best_h else best_h

        return best_h.detach().cpu(), best_acc_known.detach().cpu(), best_acc_unknown.detach().cpu()

    def plot_loss(self, filename):

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(self.log_loss, label='loss', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Acc', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.log_acc_known, label='S', color=color)
        ax2.plot(self.log_acc_unknown, label='U', color='tab:green')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.legend(loc='lower left', bbox_to_anchor=(1, 0.01))
        plt.savefig(filename)
        plt.clf()
