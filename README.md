This repository utilizes https://github.com/wesboyt/poker-table/blob/main/hand.py a pokerkit wrapper that interfaces with the universal poker hand format https://github.com/wesboyt/wesboyt.github.io/blob/master/UniversalPokerHandDataFormat.txt.

I am not sharing my pretraining transformer models that this library refines or the encoder package that translates universal hand histories into a format a onehot tokenizer can process. However you can find my training script here: https://github.com/wesboyt/loss_scaling/blob/main/loss_scaling.py

You will need to create your own poker game encoding representation, model designs, and tokenizer vocabulary. I have an example repo that will make creating your own tokenizer language clearer: https://github.com/wesboyt/huggingface_onehot_tokenizer_generator.

This repository has a number of implementations, all implementations use a similar counter factual montecarlo self sampling. Similar to playing oneself in chess.

adv weighting --> kl divergence steers action spaces towards a target of sampled ev decisions clipped to +/- 5% of logit magnitude

soft q learning --> kl divergence updates weights to a target of softmax over sampled evs

optimal transport --> Sinkhorn updates loss based on distribution of action space sampled outcomes vs best decision outcomes. Imagine moving piles of dirt between sections of an american football field. how much work was necessary to move the dirt models outcome space to best option outcome space.

