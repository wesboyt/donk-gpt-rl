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

    def rl(self):
        losses = []
        shift_cap = 0.05
        ttl_loss = 0
        batch_size = 16
        accumulation_steps = 4

        for itr in range(0, 50000):
            batch = []
            batch_states = []
            players = []

            for i in range(batch_size):
                player = randint(0, 5)
                players.append(player)
                with torch.no_grad():
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
                    uh = hand.get_u_hand(player)
                    print(uh)
                    batch_states.append(states)
                    batch.append(self.encoder.encode(json.dumps(uh), True))

            batch_tensor = torch.tensor(self.tokenizer(batch, padding="max_length", max_length=128).input_ids).to(self.device)
            hero_ids = batch_tensor[:, 1]
            state_indexes = torch.zeros(batch_tensor.shape[0], dtype=torch.int).to(self.device).detach()
            last_hero_tokens = torch.tensor([]).to(self.device)

            current_batch_loss = torch.tensor(0.0).to(self.device)
            valid_loss_steps = 0  # Count how many tokens we actually trained on
            for i in range(1, 128):
                tokens = batch_tensor[:, i]
                subslice = batch_tensor[:, :i]
                logits = self.new_model(subslice).logits[:, -1, :]
                base_log_probs = torch.log_softmax(logits, dim=1)
                hero_tokens = torch.argwhere(hero_ids == tokens).squeeze()
                train_mask = torch.zeros(batch_tensor.shape[0], dtype=torch.bool).to(self.device)
                if i > 26:
                    action_token_indexes = torch.argwhere((tokens <= 13) & (tokens >= 9)).squeeze()

                    if action_token_indexes.ndim == 0 and action_token_indexes.numel() == 1:
                        action_token_indexes = action_token_indexes.unsqueeze(0)
                    elif action_token_indexes.numel() == 0:
                        action_token_indexes = torch.tensor([]).to(self.device)
                    if last_hero_tokens.numel() > 0:
                        hero_action_indexes = action_token_indexes[torch.isin(action_token_indexes, last_hero_tokens)]
                    else:
                        hero_action_indexes = torch.tensor([], dtype=torch.long).to(self.device)
                    target_log_probs = base_log_probs.clone().detach()

                    for index in hero_action_indexes:
                        train_mask[index] = True
                        evs = self.generate_action_evs(batch_states[index][state_indexes[index]])
                        probs = torch.exp(target_log_probs[index])
                        max_ev = 0
                        for ev_token in evs.keys():
                            abs_ev = abs(evs[ev_token])
                            if abs_ev > max_ev: max_ev = abs_ev

                        keys = list(evs.keys())
                        key_len = len(keys)
                        ttl = 0
                        temp_evs = {}
                        for ev_token in keys:
                            adj = evs[ev_token] / max_ev if max_ev != 0 else 0
                            temp_evs[ev_token] = adj
                            ttl += adj

                        mean = ttl / key_len

                        for ev_token in keys:
                            adj_ev = temp_evs[ev_token] - mean
                            factor = 1.0 + (shift_cap * adj_ev)
                            factor = max(factor, 1e-6)
                            probs[ev_token] = probs[ev_token] * factor
                        probs = probs / torch.sum(probs)
                        target_log_probs[index] = torch.log(probs + 1e-9)  # Epsilon for safety
                    player_tokens = torch.argwhere((tokens <= 28) & (tokens >= 17)).squeeze()
                    if player_tokens.numel() > 0:
                        state_indexes[player_tokens] += 1
                    if train_mask.any():
                        loss = self.loss(base_log_probs[train_mask], target_log_probs[train_mask])
                        current_batch_loss += loss
                        valid_loss_steps += 1
                        losses.append(loss.item())
                last_hero_tokens = hero_tokens
            if valid_loss_steps > 0:
                final_loss = current_batch_loss / valid_loss_steps
                final_loss.backward()
                if (itr + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                mean_loss = np.mean(losses) if losses else 0
                print(f"Itr {itr} | Mean Loss: {mean_loss:.6f}")
                losses = []
            if itr % 100 == 0 and itr > 0:
                torch.save(self.model.state_dict(), f"RL-{itr}.pt")

    """
    hard_token_indexes = torch.argwhere((tokens < 9) | ((tokens < self.min_size_token) | (tokens > 13))).squeeze()
    for index in hard_token_indexes:
        target_log_probs[index][tokens[index]] += max(math.log(1.0 + shift_cap), 1e-6)
        # --- RE-NORMALIZE IN LOG SPACE ---
        # Crucial: Adjusting individual tokens breaks the sum=1 property.
        # Log(P / Sum(P)) = Log(P) - LogSumExp(Log(P))
        row_log_sum = torch.logsumexp(target_log_probs[index], dim=0)
        target_log_probs[index] = target_log_probs[index] - row_log_sum
    """

    def get_action_likelihoods(self, hnd):
        with torch.no_grad():
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
                #if chance < 0.01:
                #    chance = 0.01
                target[self.fold_token] = chance
            if 'check' in action_space:
                chance = likelihoods[self.check_token]
                #if chance < 0.01:
                #    chance = 0.01
                target[self.check_token] = chance
            if 'call' in action_space:
                chance = likelihoods[self.call_token]
                #if chance < 0.01:
                #    chance = 0.01
                target[self.call_token] = chance
            if 'min_bet' in action_space:
                chance = likelihoods[self.raise_token]
                #if chance < 0.03:
                #    chance = 0.03
                target[self.raise_token] = chance
            effective_stack = hnd.state.get_effective_stack(hnd.state.turn_index)
            if ('call' in action_space and effective_stack == hnd.state.checking_or_calling_amount) or 'min_bet' in action_space and action_space['min_bet'] >= .25 * action_space['max_bet']:
                chance = likelihoods[self.allin_token]
                #if chance < 0.005:
                #    chance = 0.005
                target[self.allin_token] = chance
            target = target / target.sum()
            return target

    def get_raise_likelihoods(self, hnd):
        with torch.no_grad():
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

    def generate_payoffs(self, ohand, player):
        with torch.no_grad():
            payoffs = []
            for i in range(15):
                hand = copy.deepcopy(ohand)
                while not hand.done:
                    """
                    if hand.state.turn_index == player:
                        self.model = self.new_model
                    else:
                        self.model = self.original_model
                    """
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
                    if not payoff[j]:
                        payoff[j] = (-50) * hand.big_blind
                    payoff[j] /= hand.big_blind
                """
                payoffs.append(payoff)
            return payoffs

    def generate_action_evs(self, ohand):
        with torch.no_grad():
            player = ohand.state.turn_index
            results = {}
            for option in ohand.get_action_space():
                pre_hand = copy.deepcopy(ohand)
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
                    case _:
                        continue
                payoffs = self.generate_payoffs(pre_hand, player)
                results[option_token] = np.mean(list(map(lambda x: x[player], payoffs)))
            return results

    """
    def generate_raise_evs(self, ohand):
        with torch.no_grad():
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
    """



    def select_action(self, hnd):
        with torch.no_grad():
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
            if 'max_bet' in action_space:
                return 'raise', action_space['max_bet']
            else:
                return 'call', hnd.state.checking_or_calling_amount

    def select_raise(self, hand):
        with torch.no_grad():
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

