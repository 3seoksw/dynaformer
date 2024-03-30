import torch

from torch import nn
from .basemodel import BaseModel


class Dynaformer(BaseModel):
    """
    Input: context (current, voltage, time) & current
    Output:
        Encoder: Ageing inference (predict degradation parameters)
        Decoder: Voltage trajectory according to the input current

    Embedding
      |
      v
    Positional encoding
      |
      v
    Encoder
      |
      v
    Decoder
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        hidden_dim=128,
        lr=1e-3,
        patience_lr_plateau=100,
        loss="rmse",
    ):
        super().__init__()

        self.intput_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.patience_lr_plateau = patience_lr_plateau
        self.loss = loss
        self.chunks = self.hidden_dim // 2

        self.loss_func = torch.nn.MSELoss(reduction="mean")

        # Pre-encoder (linearization, embedding, positional encoding)
        self.pre_encoder_linear = nn.Linear(input_dim, hidden_dim)
        self.position_embedding_context = nn.Embedding(1000, self.hidden_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)

        # Pre-decoder
        self.step_up_query = nn.Sequential(nn.Linear(self.chunks, hidden_dim))
        self.last_query = nn.Linear(hidden_dim // 2, hidden_dim)
        self.positional_embedding_query = nn.Embedding(
            num_embeddings=450, embedding_dim=self.hidden_dim
        )

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=6)
        self.pre_fin = nn.Linear(self.hidden_dim, self.chunks)

        self.save_hyperparameters()

    def embedding(self, context):
        # NOTE: shape is in the form of:
        #       [batch_size, length, features=[voltage, current, time]]
        context_value = context[:, :, :2]
        time = context[:, :, 2]

        # Embedding & positional encoding
        src_padding_mask = torch.zeros(
            context_value.shape[0], context_value.shape[1], device=context_value.device
        )
        src_padding_mask[(context_value == 0).all(axis=2)] = 1
        positional_encoding = self.position_embedding_context(time.long())

        # Linearization
        pre_encoder = self.pre_encoder_linear(context_value)

        # Encoding
        encoder = pre_encoder + positional_encoding
        encoder = self.encoder(
            encoder.permute(1, 0, 2), src_key_padding_mask=src_padding_mask.bool()
        )  # NOTE: PyTorch encoder expects an input to be shaped as [length, batch_size, features]

        return encoder, src_padding_mask

    def forward(self, context, x):
        encoder, src_padding_mask = self.embedding(context)

        padded_x = torch.nn.functional.pad(
            x, (0, self.chunks - (x.shape[1] % self.chunks), 0, 0)
        )

        query = padded_x.reshape(
            padded_x.shape[0], padded_x.shape[1] // self.chunks, self.chunks
        )

        collate_mask = torch.zeros(query.shape[:-1], device=self.device)
        collate_mask[(query == 0).all(dim=2)] = 1.0
        all_query = self.step_up_query(query)
        query = all_query

        positions = (
            torch.arange(0, query.shape[1], device=self.device)
            .unsqueeze(0)
            .repeat(x.shape[0], 1)
        )
        positional_embedding = self.positional_embedding_query(positions)
        query = query + positional_embedding
        query = query.permute(1, 0, 2)

        output_transf = self.decoder(
            query,
            encoder,
            tgt_key_padding_mask=collate_mask.bool(),
            memory_key_padding_mask=src_padding_mask.bool(),
        )
        output_transf = output_transf.permute(1, 0, 2)
        out = self.pre_fin(output_transf)

        # restore the shape of the output
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
        assert out.shape[1] == padded_x.shape[1]

        # Remove the padding
        out = out[:, : x.shape[1]]
        return out.unsqueeze(2)
