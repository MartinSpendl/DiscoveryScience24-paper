import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from tqdm import tqdm
import time

class _VAEBase(nn.Module):
    """Implemented using this template:
    https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
    """
    def __init__(self, input_features:int, latent_features:int, hidden_layers:list, layer_activation:str="Tanh", output_type:str="mse_output", use_batch_norm:bool=True, device:str="cpu"):
        super().__init__()

        self.input_features = input_features
        self.latent_features = latent_features
        self.hidden_layers = hidden_layers
        self.layer_activation = layer_activation
        self.output_type = output_type
        self.use_batch_norm = use_batch_norm
        self.device = device

        self.valid_output_types = ["mse_output", "cox_output", "elbo_output", "div_output"]
        assert self.output_type in self.valid_output_types, f"Invalid output_type: {self.output_type}. Please choose among {self.valid_output_types}."

        # creating layers
        self.encoder = self.create_encoder()
        self.mean_layer, self.logvar_layer = self.create_mean_and_logvar_layers()
        self.decoder = self.create_decoder(self.output_type)

    def create_encoder(self):

        neurons = [self.input_features] + self.hidden_layers
        layers = []

        for i in range(len(neurons)-1):
            layers.append(nn.BatchNorm1d(neurons[i])) if self.use_batch_norm else None
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(neurons[i], neurons[i+1], bias=True))
            layers.append(getattr(nn, self.layer_activation)())
        
        return nn.Sequential(*layers)
    
    def create_mean_and_logvar_layers(self):
        hidden_features = self.hidden_layers[-1] if len(self.hidden_layers) > 0 else self.input_features
        return (
            nn.Linear(hidden_features, self.latent_features, bias=True),
            nn.Linear(hidden_features, self.latent_features, bias=True)
        )
    
    def create_decoder(self, output_type):

        if output_type == "mse_output":
            return self.create_mse_decoder()

        elif output_type == "cox_output":
            return self.create_cox_decoder()
        
        elif output_type == "elbo_output":
            return self.create_cox_decoder(use_bias=True)
        
        elif output_type == "div_output":
            return self.create_div_decoder()
                
        else:
            raise ValueError(f"Invalid output_type: {output_type}. Choose among: {self.valid_output_types}")
    
    def create_mse_decoder(self):

        neurons = [self.latent_features] + self.hidden_layers + [self.input_features]
        layers = []

        for i in range(len(neurons)-1):
            layers.append(nn.BatchNorm1d(neurons[i])) if self.use_batch_norm else None
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(neurons[i], neurons[i+1], bias=True))
            layers.append(getattr(nn, self.layer_activation)()) if i != len(neurons)-2 else None
        
        return nn.Sequential(*layers)

    def create_cox_decoder(self, use_bias=False):
        return _CoxDecoder(
            self.latent_features, self.hidden_layers[::-1], self.input_features,
            self.layer_activation, self.use_batch_norm, use_bias=use_bias)

    def create_div_decoder(self):
        return _DivDecoder(
            self.latent_features, self.hidden_layers[::-1], self.input_features,
            self.layer_activation, self.use_batch_norm)

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def forward(self, x):

        # Encode into latent space
        mean, logvar = self.encode(x)

        # Add noise
        epsilon = torch.randn_like(logvar).to(self.device)
        z = mean + logvar*epsilon
        
        # Reconstruction
        reconstruction = self.decoder(z)

        return reconstruction, mean, logvar


class _CoxDecoder(nn.Module):
    def __init__(self, latent_f, hidden_f:list, output_f:int, layer_activation:str, use_batch_norm:bool, use_bias:bool=False):
        super().__init__()

        self.latent_f = latent_f
        self.hidden_f = hidden_f
        self.output_f = output_f
        self.layer_activation = layer_activation
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias

        neurons = [self.latent_f] + self.hidden_f
        layers = []

        for i in range(len(neurons)-1):
            layers.append(nn.BatchNorm1d(neurons[i])) if self.use_batch_norm else None
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(neurons[i], neurons[i+1], bias=True))
            layers.append(getattr(nn, self.layer_activation)())
        
        self.base_decoder = nn.Sequential(*layers)

        self.risk_layer = nn.Linear(neurons[-1], self.output_f, bias=use_bias)
        self.risk_layer.name = "risk"
        self.tran_layer = nn.Linear(neurons[-1], self.output_f, bias=True)
        self.tran_layer.name = "transcription-cox"
    
    def forward(self, x):
        x = self.base_decoder(x)
        
        risk = self.risk_layer(x).unsqueeze(-1)
        trans = self.tran_layer(x).unsqueeze(-1)

        return torch.cat((risk, trans), dim=-1)

class _DivDecoder(nn.Module):
    def __init__(self, latent_f, hidden_f:list, output_f:int, layer_activation:str, use_batch_norm:bool):
        super().__init__()

        self.latent_f = latent_f
        self.hidden_f = hidden_f
        self.output_f = output_f
        self.layer_activation = layer_activation
        self.use_batch_norm = use_batch_norm

        neurons = [self.latent_f] + self.hidden_f
        layers = []

        for i in range(len(neurons)-1):
            layers.append(nn.BatchNorm1d(neurons[i])) if self.use_batch_norm else None
            layers.append(nn.Dropout(0.1))
            layers.append(nn.Linear(neurons[i], neurons[i+1], bias=True))
            layers.append(getattr(nn, self.layer_activation)())
        
        self.base_decoder = nn.Sequential(*layers)

        self.trans_layer = nn.Linear(neurons[-1], self.output_f)
        self.trans_layer.name = "transcription"
        self.decay_layer = nn.Linear(neurons[-1], self.output_f)
        self.decay_layer.name = "decay"
    
    def forward(self, x):
        x = self.base_decoder(x)
        
        trans = self.trans_layer(x)
        decay = self.decay_layer(x)

        return trans - decay

class GaussianCoxPHDecayLossTraditional(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, **kwargs):

        predicted_h = kwargs["predicted_risk"] # torch.Tensor shape (n, m)
        expressions = kwargs["expressions"] # torch.Tensor shape (n, m)
        predicted_tr = kwargs["predicted_tr"] # torch.Tensor shape (n, m)
        event_indicator = kwargs["event_indicator"] # torch.Tensor shape (n, m)
        sigma = kwargs["sigma"]

        time_indicator = (expressions - predicted_tr).exp()
        time_indicator = time_indicator / time_indicator.max(dim=0)[0] / (2*sigma* torch.sqrt(torch.Tensor([2]).to(self.device)))

        n, m = expressions.shape

        location = time_indicator.view(n, 1, m) - time_indicator.view(1, n, m)
        phi_matrix = 0.5 * (1 + torch.erf(-location))
        phi_matrix[torch.arange(n), torch.arange(n)] += 0.5

        cumsum = torch.matmul(predicted_h.T.exp().view(m, 1, n), phi_matrix.permute((2,1,0))).squeeze()
        log_cumsum = torch.log(cumsum).T
        sum_loglikelihood = torch.sum((predicted_h - log_cumsum) * event_indicator)
        return -sum_loglikelihood

class GaussianCoxPHDecayLossCorrect(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, **kwargs):

        predicted_h = kwargs["predicted_risk"] # torch.Tensor shape (n, m)
        expressions = kwargs["expressions"] # torch.Tensor shape (n, m)
        predicted_tr = kwargs["predicted_tr"] # torch.Tensor shape (n, m)
        event_indicator = kwargs["event_indicator"] # torch.Tensor shape (n, m)
        sigma = kwargs["sigma"]

        time_indicator = (expressions - predicted_tr).exp()
        time_indicator = time_indicator / time_indicator.max(dim=0)[0] / (2*sigma* torch.sqrt(torch.Tensor([2]).to(self.device)))

        n, m = expressions.shape

        location = time_indicator.view(n, 1, m) - time_indicator.view(1, n, m)
        phi_matrix = 0.5 * (1 + torch.erf(-location))
        phi_matrix[torch.arange(n), torch.arange(n)] += 0.5

        cumsum = torch.matmul(predicted_h.T.exp().view(m, 1, n), phi_matrix.permute((2,1,0))).squeeze()
        log_cumsum = torch.log(cumsum).T
        mean_loglikelihood = torch.sum((predicted_h - log_cumsum) * event_indicator)/predicted_h.shape[0]
        return -mean_loglikelihood

class MeanStdLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device    

    def forward(self, **kwargs):
        mean = kwargs["predicted_risk"]
        logvar = kwargs["predicted_tr"]
        return - 0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - logvar.exp())

class MeanSquaredError(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.loss = nn.MSELoss()
        self.device = device
    
    def forward(self, **kwargs):
        
        y_pred = kwargs["y_pred"] #torch.Tensor
        expressions = kwargs["expressions"] #torch.Tensor
        
        return self.loss(expressions, y_pred)

class ELBOLoss(nn.Module):
    def __init__(self, model_type, KLD_ratio:float=1.0, ortho_ratio:float=0.0, device:str="cpu"):
        super().__init__()
        self.model_type = model_type
        self.reconstruction_loss = {
            "mse": MeanSquaredError(device),
            "cox": GaussianCoxPHDecayLossCorrect(device),
            "elbo": MeanStdLoss(device),
            "div": MeanSquaredError(device),
        }[model_type]
        self.KLD_ratio = KLD_ratio
        self.ortho_ratio = ortho_ratio
        self.device = device

    def orthogonality(self, latent_space):

        LLT = torch.matmul(latent_space.T, latent_space)
        norm_L = torch.sqrt(torch.square(latent_space).sum(dim=0)).unsqueeze(dim=0)
        
        return torch.sum(torch.square(LLT/(norm_L * norm_L.T)))

    def forward(self, **kwargs):

        mean = kwargs["mean"]
        logvar = kwargs["logvar"]

        reconstruction = self.reconstruction_loss(**kwargs)

        KL_divergence = - 0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - logvar.exp())

        orthogonal_loss = self.orthogonality(mean)

        return reconstruction + self.KLD_ratio * KL_divergence + self.ortho_ratio * orthogonal_loss, reconstruction, KL_divergence

class TimeSigmaScheduler():
    def __init__(self, start_sigma=1e-2, end_sigma=1e-8, n_epochs=100):
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.n_epochs = n_epochs
        self.current_epoch = 0
        self.current_time_sigma = start_sigma
        self.steps = np.logspace(
            np.log10(self.start_sigma), np.log10(self.end_sigma), self.n_epochs,
        )
    
    def get_time_sigma(self):
        return self.current_time_sigma
    
    def step(self):
        self.current_epoch += 1
        self.current_time_sigma = self.steps[min(len(self.steps)-1, self.current_epoch)]


class VAE:
    def __init__(
            self,
            # Model parameters)
            model_type:str, # "mse", "cox", "elbo", "div"
            latent_features:int=2,
            hidden_layers:list=[],
            layer_activation:str="Tanh",
            use_batch_norm:bool=False,
            device:str="cpu",
            
            # Training parameters
            n_epochs:int=1000,
            batch_size:int=32,
            l2_penalty:float=0.001,
            KLD_ratio:float=1.0,
            orthogonal_ratio:float=0,
            start_time_sigma:float=1e-2,
            end_time_sigma:float=1e-8,
            n_epochs_sigma:int=100,
            censoring_ratio:float=0.0,
            early_stopping:bool=False,
            no_patience_before:int=10,
            patience:int=10,
            val_size:float = 0.2,
            cox_additional_evals:int=1,
            lr: float=0.01,
            cosine_Tmax:int=50,
            optimizer_name:str="Adam",
            verbose:int=0,
            seed:int=0,
            ):
        """
        Arguments:
        ----------
        model_type: str
            Choose among "mse", "cox" and "elbo" model type.
        latent_features: int
            Number of latent space dimensions.
        hidden_layers: list
            List with hidden layer sizes. Used in both encoder and reverse in decoder.
        layer_activation: str
            Activation function between layers
        use_batch_norm: bool
            Use batch normalization between layers.
        device: str
            Device to train the model on (default: cpu)
        
        n_epochs: int
            Maximal number of epochs.
        batch_size: int
            Number of samples in a batch.
        l2_penalty: float
            L2 penalty term added to the loss function.
        KLD_ratio: float
            Ratio between KL divergence and reconstruction loss term.
        orthogonal_ratio: float
            Ratio between reconstruction and orthogoanlity loss.
        start_time_sigma: float
            Time sigma in SSLCoxPH model at the start of the training. Ignored in other models.
        end_time_sigma: float
            Time sigma in SSLCoxPH model after n_epochs_sigma of the training. Ignored in other models.
        n_epochs_sigma: int
            The number of epochs for reducing the time sigma from start to end sigma, logarithmically. Ignored in other models.
        early_stopping: bool
            Use early stopping
        no_patience_before: int
            Ignore patience before N epochs.
        patience: int
            Patience before early stopping.
        val_size: float
            Size of the validation dataset
        cox_additional_evals: int
            Additional iterations of validation function when using SSLCoxPH.
            If batch size is smaller than validation set, perform more iterations of validation
            for more stable validation loss estimation.
        lr: float
            Initial learning rate
        cosine_Tmax: int
            Tmax parameter in CosineAlignerLR
        optimizer_name: str
            Name of the PyTorch optimizer.
        verbose: int
            Verbose >1 or silent = 0.
        seed: int
            Random seed for numpy and torch intialization.
        """
        # Model parameters
        self.model_type = model_type
        self.latent_features = latent_features
        self.hidden_layers = hidden_layers
        self.layer_activation = layer_activation
        self.use_batch_norm = use_batch_norm
        self.device = device

        # Training parameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_penalty = l2_penalty
        self.KLD_ratio = KLD_ratio
        self.orthogonal_ratio = orthogonal_ratio
        self.time_sigma = start_time_sigma
        self.start_time_sigma = start_time_sigma
        self.end_time_sigma = end_time_sigma
        self.n_epochs_sigma = n_epochs_sigma
        self.censoring_ratio = censoring_ratio
        self.early_stopping = early_stopping
        self.no_patience_before = no_patience_before
        self.patience = patience
        self.val_size = val_size
        self.cox_additional_evals = cox_additional_evals
        self.lr = lr
        self.cosine_Tmax = cosine_Tmax
        self.optimizer_name = optimizer_name
        self.verbose = verbose
        self.seed = seed

    def _assert_encoder_exists(self):
        assert hasattr(self, "model"), "Model has not yet been fitted. Call .fit() function first."

    def predict(self, X:pd.DataFrame, y=None) -> np.ndarray:
        self._assert_encoder_exists()
        self.model.eval()
        input_X = torch.from_numpy(X[self.feature_columns].values if type(X) == pd.DataFrame else X)
        input_X = input_X.type(torch.float32).to(self.device)
        y_pred = self.model(input_X)[0].detach().cpu().numpy()
        return pd.DataFrame(y_pred, index=X.index) if type(X) == pd.DataFrame else y_pred
    
    def predict_latent(self, X:pd.DataFrame):
        self._assert_encoder_exists()
        self.model.eval()
        input_X = torch.from_numpy(X[self.feature_columns].values if type(X) == pd.DataFrame else X)
        input_X = input_X.type(torch.float32).to(self.device)
        means, _ = self.model.encode(input_X)
        means = means.detach().cpu().numpy()
        return pd.DataFrame(means, index=X.index) if type(X) == pd.DataFrame else means

    def fit(self, X: pd.DataFrame) -> None:
        """
        Initiate self.encoder, self.head and train them using self._train_model() function.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.feature_columns = X.columns

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_events = self.n_samples
        
        self.model = _VAEBase(
            input_features=self.n_features,
            latent_features=self.latent_features,
            hidden_layers=self.hidden_layers,
            layer_activation=self.layer_activation,
            output_type=f"{self.model_type}_output",
            use_batch_norm=self.use_batch_norm,
            device=self.device
        )
        self.model.to(self.device)

        if self.val_size > 0:
            train_ids, test_ids = train_test_split(np.arange(X.shape[0]), test_size=self.val_size)
        else:
            train_ids, test_ids = np.arange(len(X)), []
        
        self._train_ids, self._test_ids = train_ids.copy(), test_ids.copy()
        self._train_model(X, X, train_ids, test_ids)

    def _Xy_to_tensor(self, X, y):
        return (
            torch.from_numpy(X.values).type(torch.float32).to(self.device),
            torch.from_numpy(y.values).type(torch.float32).to(self.device),
        )

    def _get_loss_kwargs(self, y_pred, y_true, mean, logvar):
        kwargs = {
                "y_pred": y_pred,
                "expressions": y_true,
                "mean": mean,
                "logvar": logvar,
                "sigma": self.time_sigma,
            }
        if self.model_type in ["cox", "elbo"]:
            kwargs.update({
                "predicted_risk": y_pred[:, :, 0],
                "predicted_tr": y_pred[:, :, 1],
                "event_indicator": torch.bernoulli(torch.empty_like(y_true).uniform_(0,1), p=1-self.censoring_ratio) ,#torch.ones_like(y_true), torch.bernoulli(torch.empty_like(y_true).uniform_(0,1), p=0.9)
                "sigma": self.time_sigma,
            })
        
        return kwargs

    def _train_model(self, X, y, train_ids, test_ids):

        # Prepare data
        X_train, y_train = X.iloc[train_ids], y.iloc[train_ids]
        X_val, y_val = X.iloc[test_ids], y.iloc[test_ids]
        X_whole, y_whole = (
            X.iloc[np.append(train_ids, test_ids)],
            y.iloc[np.append(train_ids, test_ids)],
        )

        X_train, y_train = self._Xy_to_tensor(X_train, y_train)
        X_val, y_val = self._Xy_to_tensor(X_val, y_val)
        X_whole, y_whole = self._Xy_to_tensor(X_whole, y_whole)

        # Prepare training parameters and optimizer
        param_groups = [
            {"params": [p for n, p in self.model.named_parameters() if "transcription-cox" in n], "weight_decay": 0.0},
            {"params": [p for n, p in self.model.named_parameters() if "transcription-cox" not in n], "weight_decay": self.l2_penalty},
        ]
        optimizer = getattr(torch.optim, self.optimizer_name)(param_groups, lr=self.lr)
        self.loss_function = ELBOLoss(self.model_type, self.KLD_ratio, self.orthogonal_ratio, self.device)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.cosine_Tmax)
        time_sigma_scheduler = TimeSigmaScheduler(start_sigma=self.start_time_sigma, end_sigma=self.end_time_sigma, n_epochs=self.n_epochs_sigma)

        # Training loop
        self.training_loss = []
        self.training_loss_decoupled = []
        self.validation_loss = []
        self.validation_loss_decoupled = []
        for epoch in tqdm(range(1, self.n_epochs + 1)):
            train_ids_shuffled = torch.randperm(X_train.size()[0])
            
            self.model.train()

            trl, trl_decoupled = [], []
            vll, vll_decoupled = [], []

            self.time_sigma = time_sigma_scheduler.get_time_sigma()
            
            # TRAINING
            batches = X_train.shape[0]//self.batch_size + 1 
            for j in range(batches):

                if self.model_type == 'cox':
                    train_ids_shuffled = torch.randperm(X_train.size()[0])[:self.batch_size]
                    x, y = X_train[train_ids_shuffled], y_train[train_ids_shuffled]
                else:
                    ids = train_ids_shuffled[j*self.batch_size:(j+1)*self.batch_size]
                    x, y = X_train[ids], y_train[ids]

                y_pred, mean, logvar = self.model(x)
                train_kwargs = self._get_loss_kwargs(y_pred, y, mean, logvar)
                loss, _recon, _KLD = self.loss_function(**train_kwargs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trl += [loss.detach().cpu().item()]
                trl_decoupled += [(_recon.detach().cpu().item(), _KLD.detach().cpu().item())]

            self.training_loss += [sum(trl)]
            self.training_loss_decoupled += [tuple(np.sum(np.array(trl_decoupled), axis=0)) if len(trl_decoupled) > 1 else trl_decoupled[0]]

            with torch.no_grad():
                self.model.eval()
                val_ids_shuffled = torch.randperm(X_val.size()[0])

                cox_additional_evals = 1 if self.model_type != 'cox' else self.cox_additional_evals
                batches = X_val.shape[0]//self.batch_size + cox_additional_evals
                for j in range(batches):

                    if self.model_type == 'cox':
                        val_ids_shuffled = torch.randperm(X_val.size()[0])[:self.batch_size]
                        x, y = X_val[val_ids_shuffled], y_val[val_ids_shuffled]
                    else:
                        ids = val_ids_shuffled[j*self.batch_size:(j+1)*self.batch_size]
                        x, y = X_val[ids], y_val[ids]

                    y_pred, mean, logvar = self.model(x)
                    val_kwargs = self._get_loss_kwargs(y_pred, y, mean, logvar)
                    loss_val, _recon_val, _KLD_val = self.loss_function(**val_kwargs)

                    vll += [loss_val.cpu()]
                    vll_decoupled += [(_recon_val.cpu(), _KLD_val.cpu())]

                self.validation_loss += [sum(vll)]
                self.validation_loss_decoupled += [tuple(np.sum(np.array(vll_decoupled), axis=0)) if len(vll_decoupled) > 1 else vll_decoupled[0]]

            scheduler.step()
            time_sigma_scheduler.step()

            if self.verbose > 0:
                print(
                    f"Epoch {epoch}, loss {self.training_loss[-1]:.4f}, val_loss {self.validation_loss[-1]:.4f}"
                )
            
            if (
                self.early_stopping
                and np.argmin(self.validation_loss[-min(self.patience, len(self.validation_loss)) :]) == min(self.patience, len(self.validation_loss)) - 1
            ):
                best_state_dict = self.model.state_dict().copy()

            if (
                self.early_stopping
                and epoch >= self.no_patience_before + self.patience
                and np.argmin(self.validation_loss[-self.patience :]) == 0
            ):
                print(f"Early stopping at epoch {epoch}/{self.n_epochs}.")
                self.model.load_state_dict(best_state_dict)
                break
        
        self.training_loss = np.array(self.training_loss)
        self.validation_loss = np.array(self.validation_loss)
        self.train_loss_decoupled = np.array(self.training_loss_decoupled)
        self.validation_loss_decoupled = np.array(self.validation_loss_decoupled)
    
    def evaluate(self, X_test):
        self._assert_encoder_exists()
        self.model.eval()
        input_X = torch.from_numpy(X_test[self.feature_columns].values if type(X_test) == pd.DataFrame else X_test)
        input_X = input_X.type(torch.float32).to(self.device)
        y_pred, mean, logvar = self.model(input_X)
        eval_kwargs = self._get_loss_kwargs(y_pred, input_X, mean, logvar)
        loss, recon, _ = self.loss_function(**eval_kwargs)
        return recon.item()
