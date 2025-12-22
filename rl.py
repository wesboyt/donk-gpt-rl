import copy
import time
from random import random, randint

from transformers import AutoModelForCausalLM, GPT2TokenizerFast, AutoConfig
import json

from hand import Hand
import torch
import numpy as np
from live_bc_encoder import Encoder

class Simulator:
    def __init__(self, path):
        self.device = 'cuda'
        self.original_model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained('./opt-it-2')
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.id_to_str = []
        self.result_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.fold_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.check_token = torch.tensor(self.tokenizer.encode("<x>")).to(self.device)
        self.call_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.flop_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.turn_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.river_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.raise_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.allin_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.unknown_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.equity_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.win_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.lose_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.min_size_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.max_size_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.end_token = torch.tensor([2]).to(self.device)
        self.sizes = list(range(1, 5))
        self.sizes = np.int16(self.sizes)
        self.torch_sizes = torch.tensor(self.sizes).to(self.device)
        self.torch_sizes_float = self.torch_sizes.float()
        self.encoder = Encoder()
        self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        torch.autograd.set_detect_anomaly(True)

        for i in range(86):
            self.id_to_str.append(self.tokenizer.decode(i))

    def get_action_likelihoods(self, hnd):
        action_space = hnd.get_action_space()
        uh = hnd.get_u_hand()
        encoded = self.tokenizer.encode(self.encoder.encode(json.dumps(uh)))
        input_ids = torch.tensor(encoded).to(self.device)
        input_ids = torch.cat((input_ids, input_ids[1].unsqueeze(0)), 0).unsqueeze(0)
        logits = self.model(input_ids).logits[:, -1, :][0]
        likelihoods = torch.softmax(logits, 0)
        target = torch.zeros(likelihoods.shape).to(self.device)

        if 'fold' in action_space:
            chance = likelihoods[self.fold_token]
            if chance < 0.001:
                chance = 0.001
            target[self.fold_token] = chance
        if 'check' in action_space:
            chance = likelihoods[self.check_token]
            if chance < 0.001:
                chance = 0.001
            target[self.check_token] = chance
        if 'call' in action_space:
            chance = likelihoods[self.call_token]
            if chance < 0.001:
                chance = 0.001
            target[self.call_token] = chance
        if 'min_bet' in action_space:
            chance = likelihoods[self.raise_token]
            if chance < 0.01:
                chance = 0.01
            target[self.raise_token] = chance
        effective_stack = hnd.state.get_effective_stack(hnd.state.turn_index)
        if ('call' in action_space and effective_stack == hnd.state.checking_or_calling_amount) or 'min_bet' in action_space and action_space['min_bet'] >= .25 * action_space['max_bet']:
            chance = likelihoods[self.allin_token]
            if chance < 0.001:
                chance = 0.001
            target[self.allin_token] = chance
        target = target / target.sum()
        return target

    def get_raise_likelihoods(self, hnd):
        action_space = hnd.get_action_space()
        uh = hnd.get_u_hand()
        encoded = self.tokenizer.encode(self.encoder.encode(json.dumps(uh)))
        input_ids = torch.tensor(encoded).to(self.device)
        input_ids = torch.cat((input_ids, input_ids[1].unsqueeze(0), self.raise_token), 0).unsqueeze(0)
        logits = self.model(input_ids).logits[:, -1, :][0]
        likelihoods = torch.softmax(logits, 0)
        pot_size = hnd.pot_size()
        min_bet = action_space['min_bet']
        min_bet_token = self.min_size_token
        for i, size in enumerate(self.torch_sizes_float):
            temp_size = pot_size * (size / 100)
            if temp_size >= min_bet:
                min_bet_token = self.min_size_token + i
                break

        bets = likelihoods[min_bet_token:]
        bets = bets / bets.sum()
        return bets, min_bet_token

    def generate_payoffs(self, hand):
        payoffs = []
        for i in range(5):
            hand = copy.deepcopy(hand)
            while not hand.done:
                action, size = self.select_action(hand)
                match action:
                    case 'fold':
                        hand.fold()
                    case 'check':
                        hand.check()
                    case 'call':
                        hand.call()
                    case 'raise':
                        hand.bet_or_raise(size)
            payoff = hand.state.payoffs
            """
            for j, val in enumerate(payoff):
                if val == 0:
                    payoff[j] = -1
            """
            payoffs.append(payoff)
        return payoffs

    def generate_action_evs(self, ohand):
        player = ohand.state.turn_index
        results = {}
        for option in ohand.get_action_space():
            pre_hand = copy.deepcopy(ohand)
            multiplier = 1.0
            match option:
                case 'fold':
                    pre_hand.fold()
                    option_token = self.fold_token.item()
                case 'check':
                    pre_hand.check()
                    option_token = self.check_token.item()
                case 'call':
                    pre_hand.call()
                    option_token = self.call_token.item()
                case 'min_bet':
                    size = self.select_raise(pre_hand)
                    pre_hand.bet_or_raise(size)
                    option_token = self.raise_token.item()
                    multiplier = 1.1
                case _:
                    continue
            payoffs = self.generate_payoffs(pre_hand)
            results[option_token] = np.mean(list(map(lambda x: x[player], payoffs))) * multiplier
        return results

    def generate_raise_evs(self, ohand):
        options = {}
        #keys need to be tokens not sizes.
        action_space = ohand.get_action_space()
        player = action_space['player']
        bets, min_bet_token = self.get_raise_likelihoods(ohand)

        roll = random()
        index = 0
        count = 0.0
        while count < roll:
            count += bets[index]
            index += 1
        roll_bet_token = min_bet_token + index
        if roll_bet_token > self.max_size_token:
            roll_bet_token = self.max_size_token
        pot_size = ohand.pot_size()
        bet = int(pot_size * self.torch_sizes_float[roll_bet_token - self.min_size_token] / 100)
        hand = copy.deepcopy(ohand)
        hand.bet_or_raise(bet)
        roll_payoffs = self.generate_payoffs(hand)
        roll_ev = np.mean(list(map(lambda x: x[player], roll_payoffs)))
        options[roll_bet_token.item()] = roll_ev

        random_token = randint(min_bet_token.item(), self.max_size_token)
        if random_token != roll_bet_token:
            bet = int(pot_size * self.torch_sizes_float[random_token - self.min_size_token] / 100)
            hand = copy.deepcopy(ohand)
            hand.bet_or_raise(bet)
            random_payoffs = self.generate_payoffs(hand)
            random_ev = np.mean(list(map(lambda x: x[player], random_payoffs)))
            options[random_token] = random_ev
        #bet = int(pot_size * self.torch_sizes_float[min_bet_token - self.min_size_token + index] / 100)
        return options


    def rl(self):
        losses = []
        shift_cap = 0.2
        ttl_loss = torch.zeros(1).to(self.device)
        batch_size = 1
        for itr in range(0, 50000):
            batch = []
            batch_states = []
            for i in range(batch_size):
                hand = Hand()
                states = []
                while not hand.done:
                    hand_copy = copy.deepcopy(hand)
                    action, size = self.select_action(hand)
                    states.append(hand_copy)
                    match action:
                        case 'fold':
                            hand.fold()
                        case 'check':
                            hand.check()
                        case 'call':
                            hand.call()
                        case 'raise':
                            hand.bet_or_raise(size)
                print(hand.u_hand)
                for player in range(6):
                #player = randint(0, 5)
                    uh = hand.get_u_hand(player)
                    batch_states.append(states)
                    batch.append(self.encoder.encode(json.dumps(uh)))
            batch = torch.tensor(self.tokenizer(batch, padding="max_length", max_length=128).input_ids).to(self.device)
            hero_ids = batch[:, 1]
            state_indexes = torch.zeros(batch.shape[0], dtype=torch.int).to(self.device)
            last_hero_tokens = torch.tensor([])
            last_last_hero_tokens = torch.tensor([])
            last_raises = torch.tensor([])
            for i in range(1, 128):
                tokens = batch[:, i]
                subslice = batch[:, :i]
                logits = self.model(subslice).logits[:, -1, :]
                base_logits = torch.log_softmax(logits, dim=1).clone()
                target = torch.softmax(logits, dim=1).clone()
                hero_tokens = torch.argwhere(hero_ids == tokens).squeeze()
                raise_tokens = torch.argwhere(tokens == self.raise_token).squeeze()
                if i > 26:
                    action_token_indexes = torch.argwhere((tokens <= 13) & (tokens >= 9)).squeeze()
                    hero_action_indexes = action_token_indexes[torch.isin(action_token_indexes, last_hero_tokens)]
                    for index in hero_action_indexes:
                        evs = self.generate_action_evs(batch_states[index][state_indexes[index]])
                        max_ev = 0
                        for ev_token in evs.keys():
                            abs_ev = abs(evs[ev_token])
                            if abs_ev > max_ev:
                                max_ev = abs_ev
                        ttl = 0
                        for ev_token in evs.keys():
                            adj = evs[ev_token] / max_ev
                            evs[ev_token] = adj
                            ttl += adj
                        mean = ttl / len(evs.keys())
                        for ev_token in evs.keys():
                            weight = target[index][ev_token]
                            adj_ev = evs[ev_token] - mean
                            target[index][ev_token] = weight + (shift_cap * weight) * adj_ev
                        target[index] /= target[index].sum()
                        
                        
                        
                        
                    player_tokens = torch.argwhere((tokens <= 28) & (tokens >= 17)).squeeze()
                    state_indexes[player_tokens] += 1

                loss = self.loss(base_logits, target.log())
                losses.append(loss.item())
                ttl_loss += loss
                    

                last_hero_tokens = hero_tokens
                last_last_hero_tokens = last_hero_tokens
                last_raises = raise_tokens
            print(np.mean(losses))
            losses = []
            ttl_loss /= batch_size
            ttl_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            ttl_loss = torch.zeros(1).to(self.device)

    def select_action(self, hnd):
        action_space = hnd.get_action_space()
        target = self.get_action_likelihoods(hnd)
        roll = random()
        count = 0
        if 'min_bet' in action_space:
            count += target[self.raise_token].item()
            if count >= roll:
                bet = self.select_raise(hnd)
                return 'raise', bet
        if 'call' in action_space:
            count += target[self.call_token].item()
            if count >= roll:
                return 'call', hnd.state.checking_or_calling_amount
        if 'fold' in action_space:
            count += target[self.fold_token].item()
            if count >= roll:
                return 'fold', 0
        if 'check' in action_space:
            count += target[self.check_token].item()
            if count >= roll:
                return 'check', 0

        #were allin
        if 'max_bet' in action_space:
            return 'raise', action_space['max_bet']
        else:
            return 'call', hnd.state.checking_or_calling_amount

    def select_raise(self, hand):
        bets, min_bet_token = self.get_raise_likelihoods(hand)
        roll = random()
        index = 0
        count = bets[index]
        while count < roll and index < bets.shape[0]:
            index += 1
            count += bets[index]
        pot_size = hand.pot_size()
        bet = pot_size * self.torch_sizes_float[min_bet_token + index - self.min_size_token] / 100
        return int(bet)


if __name__ == '__main__':
    sim = Simulator('out/checkpoint-140000')
    sim.rl()
