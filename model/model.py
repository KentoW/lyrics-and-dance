# -*- coding: utf-8 -*-
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from typing import Optional
import torch.nn.init as init


def pad_zero(x, length):
    x_len, a = x.shape
    if x_len < length:
        padding = torch.zeros(length - x_len, a, requires_grad=True, device=x.device)
        padded_tensor = torch.cat([x, padding], dim=0)
    else:
        padded_tensor = x
    return padded_tensor

def create_attention_mask(lengths):
    max_len = max(lengths)
    batch_size = len(lengths)
    mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    for idx, length in enumerate(lengths):
        mask[idx, :length] = False
    return mask


class SkeletalEncoder(nn.Module):
    def __init__(self, dim=512, bone_size=53, skeletal_dim=12, lpe_dim=12, tf_dim=16):
        super().__init__()
        self.tf_dim = tf_dim
        self.dim = dim
        self.bone_size = bone_size
        self.skeletal_dim = skeletal_dim
        self.lpe_dim = lpe_dim
        """ Init transform for skeletal frame """
        middle_dim = (skeletal_dim+tf_dim)//2
        self.bone_weights1 = nn.Parameter(torch.randn(bone_size, skeletal_dim, middle_dim))
        self.bone_biases1 = nn.Parameter(torch.randn(bone_size, middle_dim))
        self.bone_weights2 = nn.Parameter(torch.randn(bone_size, middle_dim, tf_dim))
        self.bone_biases2 = nn.Parameter(torch.randn(bone_size, tf_dim))
        """ Init transform for positional vector """
        middle_dim = (lpe_dim+tf_dim)//2
        self.position_fc1 = nn.Linear(lpe_dim, middle_dim)
        self.position_fc2 = nn.Linear(middle_dim, tf_dim)
        """ Transformer """
        encoder_layer = nn.TransformerEncoderLayer(d_model=tf_dim, 
                                                   nhead=4, 
                                                   dim_feedforward=tf_dim*2, 
                                                   dropout=0.1, 
                                                   activation="gelu", 
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        """ Pose Encoder """
        pose_dim = bone_size*tf_dim
        self.pose_encoder = nn.Sequential(nn.Linear(pose_dim, (pose_dim+dim)//2),
                                           nn.BatchNorm1d((pose_dim+dim)//2),
                                           nn.ReLU(), 
                                           nn.Linear((pose_dim+dim)//2, dim)) 

    def forward(self, frame, lpe):
        bone_hidden = torch.einsum('bnd,ndh->bnh', frame, self.bone_weights1) + self.bone_biases1
        bone_hidden = F.relu(bone_hidden)
        bone_hidden = torch.einsum('bnd,ndh->bnh', bone_hidden, self.bone_weights2) + self.bone_biases2
        position_hidden = self.position_fc1(lpe)
        position_hidden = F.relu(position_hidden)
        position_hidden = self.position_fc2(position_hidden)
        input_hidden = bone_hidden+position_hidden
        encoded = self.transformer_encoder(input_hidden)
        bs, bone, dim = encoded.shape
        encoded = encoded.view(bs, bone*dim)
        encoded = self.pose_encoder(encoded)
        return encoded


class SkeletalDecoder(nn.Module):
    def __init__(self, dim=512, bone_size=53, skeletal_dim=12, lpe_dim=12, tf_dim=16):
        super().__init__()
        self.tf_dim = tf_dim
        self.dim = dim
        self.bone_size = bone_size
        self.skeletal_dim = skeletal_dim
        self.lpe_dim = lpe_dim
        """ Pose Decoder """
        pose_dim = bone_size*tf_dim
        self.pose_decoder = nn.Sequential(nn.Linear(dim, (pose_dim+dim)//2),
                                           nn.BatchNorm1d((pose_dim+dim)//2),
                                           nn.ReLU(), 
                                           nn.Linear((pose_dim+dim)//2, pose_dim)) 
        """ Transform for positional vector """
        middle_dim = (lpe_dim+tf_dim)//2
        self.position_fc1 = nn.Linear(lpe_dim, middle_dim)
        self.position_fc2 = nn.Linear(middle_dim, tf_dim)
        """ Transformer """
        decoder_layer = nn.TransformerEncoderLayer(d_model=tf_dim, 
                                                   nhead=8, 
                                                   dim_feedforward=tf_dim*2, 
                                                   dropout=0.1, 
                                                   activation="gelu", 
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        """ Final transform for skeletal frame """
        middle_dim = (skeletal_dim+tf_dim)//2
        self.bone_weights1 = nn.Parameter(torch.randn(bone_size, tf_dim, middle_dim))
        self.bone_biases1 = nn.Parameter(torch.randn(bone_size, middle_dim))
        self.bone_weights2 = nn.Parameter(torch.randn(bone_size, middle_dim, skeletal_dim))
        self.bone_biases2 = nn.Parameter(torch.randn(bone_size, skeletal_dim))

    def forward(self, encoded, lpe):
        encoded = self.pose_decoder(encoded)
        input_hidden = encoded.view(-1, self.bone_size, self.tf_dim)
        position_hidden = self.position_fc1(lpe)
        position_hidden = F.relu(position_hidden)
        position_hidden = self.position_fc2(position_hidden)
        input_hidden = input_hidden+position_hidden
        decoded = self.transformer_decoder(input_hidden)
        decoded = torch.einsum('bnd,ndh->bnh', decoded, self.bone_weights1) + self.bone_biases1
        decoded = F.relu(decoded)
        decoded = torch.einsum('bnd,ndh->bnh', decoded, self.bone_weights2) + self.bone_biases2
        decoded = F.tanh(decoded)
        return decoded


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=512, dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        middle_dim = (input_dim+dim)//2
        """ Feature Encoder """
        self.input_encoder = nn.Sequential(nn.Linear(input_dim, middle_dim),
                                           nn.BatchNorm1d(middle_dim),
                                           nn.ReLU(), 
                                           nn.Linear(middle_dim, dim)) 
        """ Temporal Transformer """
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, 
                                                   nhead=8, 
                                                   dim_feedforward=dim*2, 
                                                   dropout=0.1, 
                                                   activation="gelu", 
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, feats, length):
        L = max(length)
        """ encode input feature """
        ft = self.input_encoder(feats)
        ft_split = torch.split(ft, length)
        ft_padded = [pad_zero(xs, L) for xs in ft_split]
        ft_input = torch.stack(ft_padded)
        """ encode motions """
        mask = create_attention_mask(length).to(feats.device)
        motion = self.transformer_encoder(ft_input, src_key_padding_mask=mask)
        return motion


class TemporalDecoder(nn.Module):
    def __init__(self, output_dim=512, dim=512):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        """ Temporal Transformer """
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim, 
                                                   nhead=8, 
                                                   dim_feedforward=dim*2, 
                                                   dropout=0.1, 
                                                   activation="gelu", 
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)
        self.output_decoder = nn.Linear(dim, output_dim)


    def forward(self, encoded, length):
        L = max(length)
        """ padding motion """
        enc_split = torch.split(encoded, length)
        enc_padded = [pad_zero(xs, L) for xs in enc_split]
        enc_input = torch.stack(enc_padded)
        """ decode motions """
        mask = create_attention_mask(length).to(encoded.device)
        decoded = self.transformer_decoder(enc_input, src_key_padding_mask=mask)
        """ extract non zero padding decoded """
        decoded = torch.cat([decoded[i, :l] for i, l in enumerate(length)], dim=0)
        decoded = self.output_decoder(decoded)
        return decoded




class AE(nn.Module):
    def __init__(self, ske_encoder, ske_decoder, tempo_encoder, tempo_decoder, use_skeletal_feature, use_affective_feature):
        super().__init__()
        self.ske_encoder = ske_encoder
        self.ske_decoder = ske_decoder
        self.tempo_encoder = tempo_encoder
        self.tempo_decoder = tempo_decoder
        self.use_skeletal_feature = use_skeletal_feature
        self.use_affective_feature = use_affective_feature
        self.criterion = nn.MSELoss()

    def forward(self, skes, afts, lpe, length):
        """ skeletal encoder """
        if self.use_skeletal_feature == 1:
            frame_encoded = self.ske_encoder(skes, lpe)

        """ concat feature """
        if self.use_skeletal_feature == 1 and self.use_affective_feature == 1:
            frame_encoded = torch.cat([frame_encoded, afts], dim=1)
        elif self.use_affective_feature == 1:
            frame_encoded = afts

        """ temporal encoder """
        tempo_encoded = self.tempo_encoder(frame_encoded, length)
        tempo_encoded = torch.cat([tempo_encoded[i, :l] for i, l in enumerate(length)], dim=0)

        """ temporal decoder """
        tempo_decoded = self.tempo_decoder(tempo_encoded, length)

        loss = 0.0
        """ split feature """
        if self.use_skeletal_feature == 1 and self.use_affective_feature == 1:
            split_tensors = torch.split(tempo_decoded, (self.tempo_encoder.dim, self.tempo_encoder.input_dim-self.tempo_encoder.dim), dim=1)
            tempo_decoded = split_tensors[0]
            aft_output = F.tanh(split_tensors[1])
            loss = self.criterion(aft_output, afts)
        elif self.use_affective_feature == 1:
            loss = self.criterion(F.tanh(tempo_decoded), afts)
            return loss

        """ skeletal decoder """
        skeletal_decoded = self.ske_decoder(tempo_decoded, lpe)
        loss = loss + self.criterion(skeletal_decoded, skes)
        return loss

    
    def get_vecs(self, skes, afts, lpe, length):
        """ skeletal encoder """
        if self.use_skeletal_feature == 1:
            encoded = self.ske_encoder(skes, lpe)
            skeletal_encoded = encoded

        """ concat feature """
        if self.use_skeletal_feature == 1 and self.use_affective_feature == 1:
            encoded = torch.cat([encoded, afts], dim=1)
        elif self.use_affective_feature == 1:
            encoded = afts

        """ temporal encoder """
        tempo_encoded = self.tempo_encoder(encoded, length)
        tempo_encoded = torch.cat([tempo_encoded[i, :l] for i, l in enumerate(length)], dim=0)

        return skeletal_encoded, tempo_encoded

    

