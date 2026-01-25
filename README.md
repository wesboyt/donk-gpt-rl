This repository utilizes https://github.com/wesboyt/poker-table/blob/main/hand.py a pokerkit wrapper that interfaces with the universal poker hand format https://github.com/wesboyt/wesboyt.github.io/blob/master/UniversalPokerHandDataFormat.txt.

I am not sharing my pretraining transformer models that this library refines or the encoder package that translates universal hand histories into a format a onehot tokenizer processes.

You will need to create your own poker game encoding representation, model designs, and tokenizer vocabulary. I have an example repo that will make creating your own tokenizer language clearer: https://github.com/wesboyt/huggingface_onehot_tokenizer_generator.

This approach is 94% accurate to professonial player actions and 74% accurate to sizings +/- 10% on over 200m pokerstars hand histories.

This RL suite processes approximately 120k simulations per hour on a 3090 utilizing 20gb of vram.

It represents the culmination of 6000 hours of research into game theory and poker(beginning in 2019) and is able to beat the bot collusion rings online by over 4bb/100 upto 200nl.

The models involved are accessible from donkgpt.com where you can submit a handhistory with an approved account and payment to receive truly gto multiway solutions that generalize to any gametree and betting structure.
