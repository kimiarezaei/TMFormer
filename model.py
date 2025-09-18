# define CNN model
import torch
import torch.nn as nn
from torchinfo import summary
import math

from utils import Params
 

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.TransformerEncoderLayer)):
        if hasattr(module,'weight'):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity='relu')
        if hasattr(module,'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


# split each signal segment into tokens
# Get features from signal
class TokenEmbedding(nn.Module):
    def __init__(self, embedding_dim, kernel_size, stride):
        super().__init__()

        self.patchifying = nn.Conv1d(1, embedding_dim[0], kernel_size=kernel_size[0], stride=stride[0], padding=kernel_size[0]//2)   # output_shape = (batch size, no. of channels, no. of tokens)
        self.trend = nn.Conv1d(1, 1, kernel_size=kernel_size[0], stride=stride[0], padding=kernel_size[0]//2, bias=False)    # moving average to get the trend
        self.cnn = nn.Conv1d(embedding_dim[0], embedding_dim[0], kernel_size=kernel_size[1], stride=stride[1], padding='same')

        self.num_patches = 1 + (19200 + 2 * (kernel_size[0] // 2) - kernel_size[0]) // stride[0]

        # Make trend fixed and not learnable(Fill weights with average values and freeze)
        self.trend.weight.data.fill_(1.0 / kernel_size[0])
        self.trend.weight.requires_grad = False

    def forward(self, X):
        patches = self.patchifying(X)
        out_cnn = self.cnn(patches)
        out_cat = torch.cat([patches, out_cnn], dim=1)  # Concatenate both kernel outputs
        out_trend = self.trend(X)
        out = out_cat + out_trend
        return out.transpose(1, 2)        # Output shape: (batch size, no. of tokens, no. of channels)  



# define the vision transformer
class TMFormer(nn.Module):
    def __init__(self, params):
        super().__init__()
        # tensors obtained from resized matrices
        self.token_embedding = TokenEmbedding(params.embedding_dim, params.kernel_size, params.stride)           # output_size = (batch size, no. of tokens, no. of channels)

        # Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, params.embedding_dim[1]) )        # cls token is introduced as a learnable parameter
        num_tokens = self.token_embedding.num_patches + 1  # Add the cls token

        # tAPE Positional embedding
        pe = self.sinusoidal_embedding(params.embedding_dim[1], num_tokens)
        self.register_buffer("pos_embedding", pe)  # Register as a fixed tensor

        self.dropout = nn.Dropout(params.drop_out_stoch)  # prevents the model from relying too much on individual tokens (patches extracted from ECG segment) and forces it to focus on the whole signal segment(4mins).
        
        # Encoder layer 
        encoder_layer = nn.TransformerEncoderLayer(d_model=params.embedding_dim[1], nhead=params.num_heads, dropout=params.drop_out_att, dim_feedforward=int(params.embedding_dim[1]*2), activation="gelu", batch_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=params.num_blks)
        
        # Classification layer (head)
        self.head = nn.Sequential(nn.LayerNorm(params.embedding_dim[1]), nn.Linear(params.embedding_dim[1], 2))  # classify to 2 classes
        

    def sinusoidal_embedding(self, dim, num_tokens):
        ## Implement sinusoidal positional encoding for dynamic shapes
        position = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = torch.zeros(num_tokens, dim)
        pe[:, 0::2] = torch.sin((position * div_term)*(dim/num_tokens))   # Apply Sine to Even Indices
        pe[:, 1::2] = torch.cos((position * div_term)*(dim/num_tokens))      # Apply Cos to Odd Indices
        return pe.unsqueeze(0)  # add batch dimension


    def forward(self,x):
        y_token_embedding = self.token_embedding(x)
        y_concat = torch.cat((self.cls_token.expand(y_token_embedding.shape[0], -1, -1), y_token_embedding), 1)   # change the token size to the size of the batches. Then add the token class to the begining of each batch
        y_dropout = self.dropout(y_concat + self.pos_embedding)
        y_encoder = self.encoder_blocks(y_dropout)
        return self.head(y_encoder[:,0])     # give the first token (class token) to the head for classification


def main():
    torch.manual_seed(42)
    params = Params(r'parameters/my_params.json')
    sample_input = torch.randn(1, 1, params.signal_length)

    ## check TokenEmbedding output
    patch_emb = TokenEmbedding( params.embedding_dim, params.kernel_size, params.stride )
    out = patch_emb(sample_input)
    print(out)
    print(out.shape)

    ## check model output
    model = TMFormer(params)
    model.apply(init_weights)
    output = model(sample_input)                # output_shape = batch_size,  number_of_samples,  embedding dimension
    print(output.shape)
    print(summary(model.to('cuda'), input_size=(1, 1, 19200)))
    


if __name__ == '__main__':
    main() 