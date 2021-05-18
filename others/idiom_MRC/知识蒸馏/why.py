from parameters import *


class XlnetCloze(XLNetPreTrainedModel):
    def __init__(self, xlnet_config, num_choices):
        super(XlnetCloze, self).__init__(xlnet_config)
        self.num_choices = num_choices
        self.transformer = XLNetModel(xlnet_config)
        self.idiom_embedding = nn.Embedding(len(config.idiom2index), xlnet_config.hidden_size)
        self.my_fc = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(xlnet_config.hidden_size, 1)
        )
        self.apply(self.init_weights)

    def forward(self, input_ids, option_ids, token_type_ids, attention_mask, positions, tags, labels=None):

        encoded_layer, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        # encoded_layer [bs, maxseq, 768] blank_states [bs,768]
        encoded_options = self.idiom_embedding(option_ids)  # [bs, 10, 768]
        multiply_result = t.einsum('abc,ac->abc', encoded_options, blank_states)  # [bs, 10, 768]
        # ipdb.set_trace()
        logits = self.my_fc(multiply_result)
        reshaped_logits = logits.view(-1, self.num_choices)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss, reshaped_logits, labels, tags
        else:
            return reshaped_logits


def load_model(epochID, model, optimizer=None):
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
                # logger.info("name {} chile {}".format(name,child))
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'transformer') else 'transformer.')
        config.logger.info("missing keys:{}".format(missing_keys))
        config.logger.info('unexpected keys:{}'.format(unexpected_keys))
        config.logger.info('error msgs:{}'.format(error_msgs))
    else:
        model_CKPT = t.load('%sBertCloze-%d-%d.pth.tar' % (config.model_root, config.version, epochID))
        state_dict = model_CKPT['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        if optimizer is not None:
            optimizer.load_state_dict(model_CKPT['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, t.Tensor):
                        state[k] = v.cuda()
        config.logger.info("load BertCloze-%d-%d successfully" % (config.version, epochID))


if __name__ == '__main__':
    xlnet_config = XLNetConfig.from_json_file(config.xlnet_config_root)
    model = XlnetCloze(xlnet_config, num_choices=10)
    print(model.state_dict().keys())
    load_model(-1, model)
