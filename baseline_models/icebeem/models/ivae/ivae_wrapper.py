import os
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data.imca import ConditionalDataset
from .ivae_core import iVAE




def IVAE_wrapper(X, U, latent_dim, batch_size=256, max_iter=7e4, seed=0, n_layers=3, hidden_dim=20, lr=1e-3, cuda=True,
                 ckpt_folder='ivae.pt', architecture="ivae", logger=None, time_limit=None, learn_decoder_var=False):
    " args are the arguments from the main.py file"
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda:0' if cuda else 'cpu')
    # print('training on {}'.format(torch.cuda.get_device_name(device) if cuda else 'cpu'))

    # load data
    # print('Creating shuffled dataset..')
    dset = ConditionalDataset(X.astype(np.float32), U.astype(np.float32), device)
    loader_params = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(dset, shuffle=True, batch_size=batch_size, **loader_params)
    data_dim, _, aux_dim = dset.get_dims()
    N = len(dset)
    max_epochs = int(max_iter // len(train_loader) + 1)

    # define model and optimizer
    # print('Defining model and optimizer..')
    model = iVAE(latent_dim, data_dim, aux_dim, activation='lrelu', device=device,
                 n_layers=n_layers, hidden_dim=hidden_dim, architecture=architecture, learn_decoder_var=learn_decoder_var)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
    # training loop
    print("Training..")

    # timer
    if time_limit is not None:
        time_limit = time_limit * 60 * 60  # convert to seconds
        t0 = time.time()
    else:
        time_limit = np.inf
        t0 = 0

    it = 0
    model.train()
    while it < max_iter and time.time() - t0 < time_limit :
        elbo_train = 0
        epoch = it // len(train_loader) + 1
        for _, (x, u) in enumerate(train_loader):
            it += 1
            optimizer.zero_grad()
            x, u = x.to(device), u.to(device)
            elbo, z_est = model.elbo(x, u)
            elbo.mul(-1).backward()
            optimizer.step()
            elbo_train += -elbo.item()
            if logger is not None and it%100 == 0:
                metrics = {"loss_train": -elbo.item()}
                logger.log_metrics(step=it, metrics=metrics)
        elbo_train /= len(train_loader)
        scheduler.step(elbo_train)

        #print('epoch {}/{} \tloss: {}'.format(epoch, max_epochs, elbo_train))
    # save model checkpoint after training
    torch.save(model.state_dict(), os.path.join(ckpt_folder, 'ivae.pt'))

    return model
