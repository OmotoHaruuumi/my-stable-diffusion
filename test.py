import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.nn import functional as F
import logging



class SimsiamCLIP(nn.Module):
    def __init__(self,med_config="configs/med_config.json"):
        """
        med_config(str): path for the mixture of encoder-decoder model's configuration file
        """
        super(SimsiamCLIP,self).__init__()
        self.visual_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.tokenizer = init_tokenizer()
        language_coder_config = BertConfig.from_json_file(med_config)
        self.language_encoder = BertModel(config=language_coder_config,add_pooling_layer=False)
        self.language_decoder = BertLMHeadModel(config=language_coder_config)
        self.language_encoder.resize_token_embeddings(len(self.tokenizer))
        self.language_decoder.resize_token_embeddings(len(self.tokenizer)) 

        #エンコーダーとデコーダーの同じ名前の層を共有する
        tie_encoder_decoder_weights(self.language_encoder,self.language_decoder.bert,'','/attention')

    def VisualEncode(self,image_tensor):
        """"
        ViTの最終層の一つ前から値を取ってくる,CLSトークンも取得する
        image_embedはタプル形式. 最初の要素は各パッチの埋め込みベクトル[バッチ，パッチ数，埋め込みベクトルの次元]
        2つめはCLSトークンが入っていて[バッチ,埋め込みベクトルの次元]
        image_embedは画像サイズが224,ViT14の場合(224/14)^2=256パッチとCLSトークンで[バッチ数,257,768] 
        """
        self.visual_encoder.eval()
        image_features = self.visual_encoder.get_intermediate_layers(image_tensor,1,False,True,True)
        expanded_cls_token = image_features[0][1]
        image_embed=torch.cat(expanded_cls_token,image_features[0][0],dim=1)
        return image_embed


    def LanguageEncode(self,caption,mode):
        assert mode in ['text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        input_embeddings = self.tokenizer(caption,paddings="max_length",truncation=True,max_length=30,
                                        return_tensors="pt")
        language_embeddings = self.text_encoder()

    
    def forward(self,image_tensor):
        image_embed = self.VosualEncode(image_tensor)
        logit,onehot,message = self.Language_Encode(image_embed)
        language_embed = self.Language_Decode(onehot)

from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    logger = logging.getLogger(__name__) 
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key) 

