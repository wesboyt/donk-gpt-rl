This repository utilizes https://github.com/wesboyt/poker-table/blob/main/hand.py a pokerkit wrapper that interfaces with the universal poker hand format https://github.com/wesboyt/wesboyt.github.io/blob/master/UniversalPokerHandDataFormat.txt.

I am not sharing my pretraining transformer models that this library refines or the encoder package that translates universal hand histories into a format a onehot tokenizer can process.

You will need to create your own poker game encoding representation, model designs, and tokenizer vocabulary. I have an example repo that will make creating your own tokenizer language clearer: https://github.com/wesboyt/huggingface_onehot_tokenizer_generator.

This approach is 94% accurate to professonial player actions and 74% accurate to sizings +/- 10% on over 200m pokerstars hand histories.

This RL suite processes approximately 120k simulations per hour on a 3090 utilizing 20gb of vram.

It represents the culmination of 6000 hours of research into game theory and poker(beginning in 2019) and is able to beat the bot collusion rings online by over 4bb/100 upto 200nl.

The models involved are accessible from donkgpt.com where you can submit a handhistory with an approved account and payment to receive truly gto multiway solutions that generalize to any gametree and betting structure.

<img width="181" height="198" alt="image" src="https://github.com/user-attachments/assets/c4ea5a05-7302-41e3-b043-e38cf37e44a0" />

This is what output action spaces look like after 180k simulations:
<img width="584" height="468" alt="image" src="https://github.com/user-attachments/assets/9f82484c-0bc0-4652-975a-c56931d9e670" />

Here is the increase in ev verse the pretrained model after 180k simulations sbvbb:
<img width="486" height="229" alt="image" src="https://github.com/user-attachments/assets/73ba5a32-58b2-4065-8711-4e46a95479d3" />

negative means rl model winning, positive means pretrained model winning, these compare performance from the exact same position with the exact same cards.


