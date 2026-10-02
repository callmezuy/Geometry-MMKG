from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model
import pdb
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward
import math
import json
import os
import os.path as osp

class AdvMixRotatE(Model):

    def __init__(
        self,
        args,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None,
        depth_emb=None
    ):

        super(AdvMixRotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        self.args = args


        if self.args.use_pool == 1 and "MKG" not in self.args.dataset:
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img_emb = img_pool(img_emb.view(-1, 64, 64))
            img_emb = img_emb.view(img_emb.size(0), -1)

            # txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            # text_emb = txt_pool(text_emb.view(-1, 12, 64))
            # text_emb = text_emb.view(text_emb.size(0), -1)
            # self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)

        if "MKG" in self.args.dataset:
            with open('./embeddings/' + 'missing.json', 'r') as file:
                data = json.load(file)
            missing_img = data[f"{self.args.dataset}-visual.pth"]
            missing_text = data[f"{self.args.dataset}-textual.pth"]
            valid_text_emb = torch.cat([text_emb[i].unsqueeze(0) for i in range(text_emb.size(0)) if i not in missing_text], dim=0)
            valid_img_emb = torch.cat([img_emb[i].unsqueeze(0) for i in range(img_emb.size(0)) if i not in missing_img], dim=0)
            self.img_mean = torch.mean(valid_img_emb, dim=0).cuda()
            self.img_std = torch.std(valid_img_emb, dim=0).cuda()
            self.text_mean = torch.mean(valid_text_emb, dim=0).cuda()
            self.text_std = torch.std(valid_text_emb, dim=0).cuda()

            for idx in missing_text:
                text_emb[idx] = self.text_mean + self.text_std * torch.randn_like(self.text_mean)
            for idx in missing_img:
                img_emb[idx] = self.img_mean + self.img_std * torch.randn_like(self.img_mean)

        else:
            self.img_mean = torch.mean(img_emb, dim=0).cuda()
            self.img_std = torch.std(img_emb, dim=0).cuda()
            self.text_mean = torch.mean(text_emb, dim=0).cuda()
            self.text_std = torch.std(text_emb, dim=0).cuda()

        if depth_emb is not None:
            self.depth_mean = torch.mean(depth_emb, dim=0).cuda()
            self.depth_std = torch.std(depth_emb, dim=0).cuda()
            self.depth_dim = depth_emb.shape[1]
            self.depth_proj = nn.Linear(self.depth_dim, self.dim_e)
            self.depth_embeddings = nn.Embedding.from_pretrained(depth_emb).requires_grad_(False)
            self.depth_proj_2 = nn.Linear(self.dim_e, self.dim_e)
        else:
            self.depth_dim = 0
            self.depth_proj = None
            self.depth_embeddings = None
            self.depth_proj_2 = None
        
        self.use_depth = depth_emb is not None

        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        # 383-256
        self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        # 384-256
        self.text_proj = nn.Linear(self.text_dim, self.dim_e)


        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(False)
        
        
        self.ent_mean = torch.mean(self.ent_embeddings.weight.data, dim=0).cuda()
        self.ent_std = torch.std(self.ent_embeddings.weight.data, dim=0).cuda()

        
        self.img_proj_2 = nn.Linear(self.dim_e, self.dim_e)
        self.text_proj_2 = nn.Linear(self.dim_e, self.dim_e)
        self.comb_proj = nn.Linear(self.dim_e * (4 if self.use_depth else 3), self.dim_e)
        
        # assert self.args.joint_way in ["Mformer_hd_mean", "Mformer_hd_graph", "Mformer_weight", "atten_weight", "learnable_weight","cat_mlp"]
        self.fusion_layer = nn.ModuleList([BertLayer(args) for _ in range(args.num_hidden_layers)])

        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        params = torch.ones((4 if self.use_depth else 3), requires_grad=True)
        self.weight_raw = torch.nn.Parameter(params)

    def add_noise_to_embeddings(self, embeddings, mean, std, noise_ratio=0.1):
        noise_mask = torch.rand(embeddings.shape[0]) < noise_ratio
        selected_embeddings = embeddings[noise_mask]
        noise = mean + std * torch.randn_like(selected_embeddings)
        embeddings[noise_mask] = (1.0 - self.args.mask_ratio) * selected_embeddings + self.args.mask_ratio * noise
        return embeddings

    def update_entity_noise(self, embeddings, batch):
        # self.entity_noise_mask.shape: torch.Size([12842])
        # noise_mask.shape: torch.Size([33792])
        noise_mask = self.entity_noise_mask[batch]
        # selected_embeddings.shape: torch.Size([2917, 256])
        # embeddings.shape: torch.Size([33792, 256])
        selected_embeddings = embeddings[noise_mask]
        # self.entity_noise.shape: torch.Size([12842, 256])
        # entity_noise_for_batch = self.entity_noise[batch]
        embeddings[noise_mask] = (1.0 - self.args.mask_ratio) * selected_embeddings + self.args.mask_ratio * self.entity_noise[batch][noise_mask]
        
        return embeddings
    def get_joint_embeddings(self, es, ev, et, ed=None):
        if ed is not None:
            e = torch.stack((es, ev, et, ed), dim=1)
        else:
            e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        hidden_states = u
        
        if "Mformer" in self.args.joint_way:
            for i, layer_module in enumerate(self.fusion_layer):
                layer_outputs = layer_module(hidden_states, output_attentions=True)
                hidden_states = layer_outputs[0]
            # torch.Size([30355, 5, 4, 4])
            if "mean" in self.args.joint_way:
                context_vectors = torch.mean(hidden_states, dim=1)
                # context_vectors = F.normalize(context_vectors)

            elif "graph" in self.args.joint_way:
                context_vectors = hidden_states[:, 0, :]
            
            elif "weight" in self.args.joint_way:
                attention_pro = torch.sum(layer_outputs[1], dim=-3)
                attention_pro_comb = torch.sum(attention_pro, dim=-2) / math.sqrt(3 * self.args.num_attention_heads)
                attention_weights = F.softmax(attention_pro_comb, dim=-1)
                context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
                # context_vectors = F.normalize(context_vectors)

        elif "atten_weight" in self.args.joint_way:
            # Attention
            scores = self.ent_attn(u).squeeze(-1)
            # attention_weights: torch.Size([33792, 3])
            attention_weights = torch.softmax(scores, dim=-1)
            # torch.Size([33792, 256])
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)

        # weight_norm_fz = F.softmax(self.weight_raw, dim=0)
        elif "learnable_weight" in self.args.joint_way:
            weight_norm_fz = F.softmax(self.weight_raw, dim=0)
            attention_weights = weight_norm_fz.unsqueeze(0).unsqueeze(-1)
            context_vectors = torch.sum(attention_weights * e, dim=1)
        else:
            if ed is not None:
                combined = torch.cat((es, ev, et, ed), dim=1)
            else:
                combined = torch.cat((es, ev, et), dim=1)
            context_vectors = self.comb_proj(combined)

        return context_vectors

    def cal_score(self, embs):
        return self._calc(embs[0], embs[2], embs[1], "")

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()


    def update_noise(self):
        text_noisy_weights = self.add_noise_to_embeddings(self.text_embeddings.weight.data.clone(), self.text_mean, self.text_std, noise_ratio=self.args.noise_ratio)
        self.text_embeddings_noise = nn.Embedding.from_pretrained(text_noisy_weights).requires_grad_(False)
        
        img_noisy_weights = self.add_noise_to_embeddings(self.img_embeddings.weight.data.clone(), self.img_mean, self.img_std, noise_ratio=self.args.noise_ratio)
        self.img_embeddings_noise = nn.Embedding.from_pretrained(img_noisy_weights).requires_grad_(False)
        
        if self.use_depth:
            depth_noisy_weights = self.add_noise_to_embeddings(self.depth_embeddings.weight.data.clone(), self.depth_mean, self.depth_std, noise_ratio=self.args.noise_ratio)
            self.depth_embeddings_noise = nn.Embedding.from_pretrained(depth_noisy_weights).requires_grad_(False)
        
        self.ent_mean = torch.mean(self.ent_embeddings.weight.data, dim=0)
        self.ent_std = torch.std(self.ent_embeddings.weight.data, dim=0)
        self.entity_noise = self.ent_mean + self.ent_std * torch.randn_like(self.ent_embeddings.weight.data)
        self.entity_noise_mask = (torch.rand(self.ent_embeddings.weight.shape[0]) < self.args.noise_ratio).cuda()

    
    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        

        if self.args.add_noise == 1 and self.img_proj.training:
            if self.args.noise_update == "epoch":
                h_img = self.img_embeddings_noise(batch_h)
                t_img = self.img_embeddings_noise(batch_t)
                h_text = self.text_embeddings_noise(batch_h)
                t_text = self.text_embeddings_noise(batch_t)
                
                if self.use_depth:
                    h_depth = self.depth_embeddings_noise(batch_h)
                    t_depth = self.depth_embeddings_noise(batch_t)
                else:
                    h_depth, t_depth = None, None
                
                h = self.update_entity_noise(h, batch_h)
                t = self.update_entity_noise(t, batch_t)
            else:
                h_img = self.img_embeddings(batch_h)
                t_img = self.img_embeddings(batch_t)
                h_text = self.text_embeddings(batch_h)
                t_text = self.text_embeddings(batch_t)
                
                self.ent_mean = torch.mean(self.ent_embeddings.weight.data, dim=0)
                self.ent_std = torch.std(self.ent_embeddings.weight.data, dim=0)
                h_img = self.add_noise_to_embeddings(h_img, self.img_mean, self.img_std, noise_ratio=self.args.noise_ratio)
                t_img = self.add_noise_to_embeddings(t_img, self.img_mean, self.img_std, noise_ratio=self.args.noise_ratio)
                h_text = self.add_noise_to_embeddings(h_text, self.text_mean, self.text_std, noise_ratio=self.args.noise_ratio)
                t_text = self.add_noise_to_embeddings(t_text, self.text_mean, self.text_std, noise_ratio=self.args.noise_ratio)
                
                if self.use_depth:
                    h_depth = self.depth_embeddings(batch_h)
                    t_depth = self.depth_embeddings(batch_t)
                    h_depth = self.add_noise_to_embeddings(h_depth, self.depth_mean, self.depth_std, noise_ratio=self.args.noise_ratio)
                    t_depth = self.add_noise_to_embeddings(t_depth, self.depth_mean, self.depth_std, noise_ratio=self.args.noise_ratio)
                else:
                    h_depth, t_depth = None, None
                
                h = self.add_noise_to_embeddings(h, self.ent_mean, self.ent_std, noise_ratio=self.args.noise_ratio)
                t = self.add_noise_to_embeddings(t, self.ent_mean, self.ent_std, noise_ratio=self.args.noise_ratio)
                    
            
        else:
            h_img = self.img_embeddings(batch_h)
            t_img = self.img_embeddings(batch_t)
            h_text = self.text_embeddings(batch_h)
            t_text = self.text_embeddings(batch_t)
            
            if self.use_depth:
                h_depth = self.depth_embeddings(batch_h)
                t_depth = self.depth_embeddings(batch_t)
            else:
                h_depth, t_depth = None, None
        
        if self.args.num_proj == 2:
            # torch.Size([33792, 256])
            h_img_emb = self.img_proj_2(self.img_proj(h_img))
            t_img_emb = self.img_proj_2(self.img_proj(t_img))
            # torch.Size([33792, 256])
            h_text_emb = self.text_proj_2(self.text_proj(h_text))
            t_text_emb = self.text_proj_2(self.text_proj(t_text))
            
            if self.use_depth:
                h_depth_emb = self.depth_proj_2(self.depth_proj(h_depth))
                t_depth_emb = self.depth_proj_2(self.depth_proj(t_depth))
            else:
                h_depth_emb, t_depth_emb = None, None
        else:
            # torch.Size([33792, 256])
            h_img_emb = self.img_proj(h_img)
            t_img_emb = self.img_proj(t_img)
            # torch.Size([33792, 256])
            h_text_emb = self.text_proj(h_text)
            t_text_emb = self.text_proj(t_text)
            
            if self.use_depth:
                h_depth_emb = self.depth_proj(h_depth)
                t_depth_emb = self.depth_proj(t_depth)
            else:
                h_depth_emb, t_depth_emb = None, None
        
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, h_depth_emb)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, t_depth_emb)
        
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score


    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)

    def get_batch_vis_embs(self, data):
        return self.img_proj(self.img_embeddings(data))

    def get_batch_text_embs(self, data):
        return self.text_proj(self.text_embeddings(data))

    def get_batch_ent_multimodal_embs(self, data):
        return self.ent_embeddings(data), self.img_proj(self.img_embeddings(data)), self.text_proj(self.text_embeddings(data))

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [8, 8, 3, 256]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # return x
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        # [8, 3, 8, 256]
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # # [8, 3, 8, 8]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [8, 3, 8, 8]
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [8, 8, 768]
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # attention: torch.Size([30355, 5, 4, 4])

        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = 0
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        if self.config.use_intermediate:
            self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )
        if not self.config.use_intermediate:
            return (self_attention_outputs[0], self_attention_outputs[1])

        attention_output = self_attention_outputs[0]
        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1]
        # present_key_value = self_attention_outputs[-1]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output, outputs)
        # if decoder, return the attn key/values as the last output
        # outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
        # return attention_output
