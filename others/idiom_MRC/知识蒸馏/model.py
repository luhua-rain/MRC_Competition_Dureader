from parameters import *

# 模型，使用xlnet编码之后进行匹配
class XlnetCloze(XLNetPreTrainedModel):
    def __init__(self, xlnet_config):
        super(XlnetCloze, self).__init__(xlnet_config)
        self.num_choices = 10
        self.transformer = XLNetModel(xlnet_config)
        self.idiom_embedding = nn.Embedding(len(config.idiom2index), xlnet_config.hidden_size)
        self.my_fc = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(xlnet_config.hidden_size, 1)
        )
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask, positions, option_ids, tags, labels=None):
        # position = tokens.index("<mask>")
        encoded_layer, _ = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        # encoded_layer [bs, maxseq, 768] blank_states [bs,768]
        encoded_options = self.idiom_embedding(option_ids)  # [bs, 10, 768]
        multiply_result = t.einsum('abc,ac->abc', encoded_options, blank_states)  # [bs, 10, 768]
        # ipdb.set_trace()
        logits = self.my_fc(multiply_result)
        reshaped_logits = logits.view(-1, self.num_choices) # [bs, 10]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss, reshaped_logits
        else:
            return reshaped_logits


# 加载模型
def load_model(epochID, model):
    if epochID == -1:
        config.logger.info('load init xlnet weight')
        state_dict = t.load(config.pretrained_xlnet_root, map_location='cpu')
        missing_keys, unexpected_keys, error_msgs = [], [], []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'transformer') else 'transformer.')
        config.logger.info("missing keys:{}".format(missing_keys))
        config.logger.info('unexpected keys:{}'.format(unexpected_keys))
        config.logger.info('error msgs:{}'.format(error_msgs))
    else:
        model_CKPT = t.load('%sXlnetCloze-%d-%d.pth.tar' % (config.model_root, config.version, epochID))
        state_dict = model_CKPT['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        config.logger.info('load %s/XlnetCloze-%d-%d.pth.tar' % (config.model_root, config.version, epochID))


# 存储模型
def save_model(epochID, model):
    t.save({'epoch': epochID,
            'state_dict': model.state_dict()},
           '%s/XlnetCloze-%d-%d.pth.tar' % (config.model_root, config.version, epochID))
    config.logger.info('save %s/XlnetCloze-%d-%d.pth.tar' % (config.model_root, config.version, epochID))


if __name__ == '__main__':
    # xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
    # model = XlnetCloze(xlnet_config, num_choices=10)
    # load_model(-1, model)
    s = "最近十年间，虚拟货币的发展可谓#idiom000381#。美国著名经济学家林顿·拉鲁什曾预言：到2050年，基于网络的虚拟货币将在某种程度上得到官方承认，成为能够流通的货币。现在看来，这一断言似乎还嫌过于保守……"
    print(config.tokenizer.tokenize(s))
