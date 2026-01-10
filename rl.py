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

    def select_action_batch(self, hands):
        with torch.no_grad():
            batch_size = len(hands)
            if batch_size == 0: return []
            batch_strs = [self.encoder.encode(json.dumps(h.get_u_hand())) for h in hands]
            batch_tokens = self.tokenizer(batch_strs, padding=True, return_tensors='pt').input_ids.to(self.device)
            logits = self.model(batch_tokens).logits[:, -1, :]
            probs = torch.softmax(logits, dim=1)
            actions = [None] * batch_size
            raise_indices = []
            fold_probs = probs[:, self.fold_token].cpu()
            check_probs = probs[:, self.check_token].cpu()
            call_probs = probs[:, self.call_token].cpu()
            raise_probs = probs[:, self.raise_token].cpu()
            allin_probs = probs[:, self.allin_token].cpu()
            for i, hnd in enumerate(hands):
                action_space = hnd.get_action_space()
                roll = random()
                cumulative = 0.0
                selected = False
                valid_probs = {}
                if 'min_bet' in action_space:
                    valid_probs['raise'] = raise_probs[i].item()
                    valid_probs['allin'] = allin_probs[i].item()
                if 'call' in action_space: valid_probs['call'] = call_probs[i].item()
                if 'fold' in action_space: valid_probs['fold'] = fold_probs[i].item()
                if 'check' in action_space: valid_probs['check'] = check_probs[i].item()

                total_p = sum(valid_probs.values())
                if total_p == 0: total_p = 1e-9

                # Selection Loop
                # Check Raise
                if 'raise' in valid_probs:
                    prob = valid_probs['raise'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        actions[i] = ('raise', 0)  # Placeholder size
                        raise_indices.append(i)
                        selected = True

                # Check Call
                if not selected and 'call' in valid_probs:
                    prob = valid_probs['call'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        # Logic for effective stack shoving
                        eff_stack = hnd.state.get_effective_stack(hnd.state.turn_index)
                        is_shove = False
                        # Simple heuristic check for all-in call based on your original code
                        if eff_stack == hnd.state.checking_or_calling_amount:
                            # Check if model wants to all-in token (contextual)
                            # Kept simple here to preserve flow
                            pass
                        actions[i] = ('call', hnd.state.checking_or_calling_amount)
                        selected = True

                # Check Fold
                if not selected and 'fold' in valid_probs:
                    prob = valid_probs['fold'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        actions[i] = ('fold', 0)
                        selected = True

                # Check Check
                if not selected and 'check' in valid_probs:
                    prob = valid_probs['check'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        actions[i] = ('check', 0)
                        selected = True

                # Fallback
                if not selected:
                    if 'max_bet' in action_space:
                        actions[i] = ('raise', action_space['max_bet'])
                    else:
                        actions[i] = ('call', hnd.state.checking_or_calling_amount)
            if raise_indices:
                raiser_tokens = batch_tokens[raise_indices]
                raise_suffix = torch.full((len(raise_indices), 1), self.raise_token.item(), device=self.device)
                raise_inputs = torch.cat([raiser_tokens, raise_suffix], dim=1)
                raise_logits = self.model(raise_inputs).logits[:, -1, :]
                raise_probs = torch.softmax(raise_logits, dim=1)
                min_bet_idx = self.min_size_token.item()
                size_probs = raise_probs[:, min_bet_idx: min_bet_idx + len(self.sizes)]
                size_probs = size_probs / (size_probs.sum(dim=1, keepdim=True) + 1e-9)
                size_probs = size_probs.cpu().numpy()

                for idx_in_subset, original_idx in enumerate(raise_indices):
                    hnd = hands[original_idx]
                    action_space = hnd.get_action_space()
                    pot_size = hnd.pot_size()
                    min_bet = action_space.get('min_bet', 0)
                    roll_size = random()
                    c_size = 0
                    chosen_pct_idx = 0
                    current_probs = size_probs[idx_in_subset]
                    for s_i, p in enumerate(current_probs):
                        c_size += p
                        if c_size >= roll_size:
                            chosen_pct_idx = s_i
                            break
                    pct = self.sizes[chosen_pct_idx]
                    bet_amount = int(pot_size * (pct / 100))
                    if bet_amount < min_bet: bet_amount = min_bet
                    if 'max_bet' in action_space and bet_amount > action_space['max_bet']:
                        bet_amount = action_space['max_bet']
                    actions[original_idx] = ('raise', bet_amount)
            return actions

    def simulate_batch(self, hands):
        active_indices = list(range(len(hands)))
        while active_indices:
            current_hands = [hands[i] for i in active_indices]
            decisions = self.select_action_batch(current_hands)
            next_active = []
            for i, (action, size) in enumerate(decisions):
                h_idx = active_indices[i]
                hnd = hands[h_idx]
                if action == 'fold':
                    hnd.fold()
                elif action == 'check':
                    hnd.check()
                elif action == 'call':
                    hnd.call()
                elif action == 'raise':
                    hnd.bet_or_raise(size)
                if not hnd.done:
                    next_active.append(h_idx)
            active_indices = next_active

    def generate_action_evs(self, ohand):
        player = ohand.state.turn_index
        action_space = ohand.get_action_space()
        scenarios = []
        n_sims = 32
        def setup_sim(action_type):
            for _ in range(n_sims):
                h = copy.deepcopy(ohand)
                h.shuffle()
                if action_type == 'fold':
                    h.fold()
                elif action_type == 'check':
                    h.check()
                elif action_type == 'call':
                    h.call()
                elif action_type == 'raise':
                    size = self.select_raise(h)
                    h.bet_or_raise(size)
                if not h.done:
                    scenarios.append(h)
                else:
                    scenarios.append(h)
        tokens_map = {}
        if 'fold' in action_space:
            start_idx = len(scenarios)
            setup_sim('fold')
            tokens_map[self.fold_token.item()] = list(range(start_idx, len(scenarios)))
        if 'check' in action_space:
            start_idx = len(scenarios)
            setup_sim('check')
            tokens_map[self.check_token.item()] = list(range(start_idx, len(scenarios)))
        if 'call' in action_space:
            start_idx = len(scenarios)
            setup_sim('call')
            tokens_map[self.call_token.item()] = list(range(start_idx, len(scenarios)))
        if 'min_bet' in action_space:
            start_idx = len(scenarios)
            setup_sim('raise')
            tokens_map[self.raise_token.item()] = list(range(start_idx, len(scenarios)))
        active_hands = [h for h in scenarios if not h.done]
        if active_hands:
            self.simulate_batch(active_hands)
        results = {}
        for token, indices in tokens_map.items():
            payoff_sum = 0
            count = 0
            for idx in indices:
                h = scenarios[idx]
                p = h.state.payoffs[player]
                # Apply tax logic
                if p > 0:
                    tax = min(p * .05, 2 * h.big_blind)
                    p -= tax
                payoff_sum += p
                count += 1
            if count > 0:
                results[token] = payoff_sum / count
        return results

    def select_raise(self, hand):
        with torch.no_grad():
            bets, min_bet_token = self.get_raise_likelihoods(hand)
            roll = random()
            index = 0
            if index < bets.shape[0]:
                count = bets[index]
                while count < roll and index < bets.shape[0] - 1:
                    index += 1
                    count += bets[index]
            pot_size = hand.pot_size()
            bet = pot_size * self.torch_sizes_float[min_bet_token + index - self.min_size_token] / 100
            return int(bet)

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

    def select_action(self, hnd):
        result = self.select_action_batch([hnd])[0]
        return result

    def rl(self):
        losses = []
        shift_cap = 0.25
        batch_size = 64
        for itr in range(0, 5000000):
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
                        hand_copy.shuffle()
                        action, size = self.select_action(hand)
                        states.append(hand_copy)
                        if action == 'fold':
                            hand.fold()
                        elif action == 'check':
                            hand.check()
                        elif action == 'call':
                            hand.call()
                        elif action == 'raise':
                            hand.bet_or_raise(size)
                    uh = hand.get_u_hand(player)
                    batch_states.append(states)
                    batch.append(self.encoder.encode(json.dumps(uh), True))

            batch_tensor = torch.tensor(self.tokenizer(batch, padding="max_length", max_length=128).input_ids).to(self.device)
            hero_ids = batch_tensor[:, 1]
            state_indexes = torch.zeros(batch_tensor.shape[0], dtype=torch.int).to(self.device).detach()
            tokens = batch_tensor[:, 0]
            last_hero_tokens = torch.argwhere(hero_ids == tokens).squeeze()
            current_batch_loss = torch.tensor(0.0).to(self.device)
            batch_logits = self.new_model(batch_tensor).logits
            for i in range(1, 128):
                tokens = batch_tensor[:, i]
                logits = batch_logits[:, i - 1, :]
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
                    if hero_action_indexes.numel() > 0:
                        for index in hero_action_indexes:
                            hero_state = batch_states[index][state_indexes[index]]
                            hero = hero_state.state.turn_index
                            train_mask[index] = True
                            evs = self.generate_action_evs(hero_state)
                            probs = torch.exp(target_log_probs[index])
                            max_ev = 0
                            for ev_token in evs.keys():
                                abs_ev = abs(evs[ev_token])
                                if abs_ev > max_ev: max_ev = abs_ev

                            keys = list(evs.keys())
                            mean = 0
                            pcts = probs.softmax(dim=0)
                            ttl_pct = 0
                            for ev_token in keys:
                                ttl_pct += pcts[ev_token].item()
                            for ev_token in keys:
                                mean += evs[ev_token] * (pcts[ev_token] / ttl_pct)
                            for ev_token in keys:
                                adj_ev = (evs[ev_token] - mean) / hero_state.pot_size()
                                factor = 1.0 + (shift_cap * adj_ev)
                                factor = max(factor, 1e-6)
                                probs[ev_token] = probs[ev_token] * factor
                            probs = probs / torch.sum(probs)
                            target_log_probs[index] = torch.log(probs + 1e-9)
                    player_tokens = torch.argwhere((tokens <= 28) & (tokens >= 17)).squeeze()
                    if player_tokens.numel() > 0:
                        state_indexes[player_tokens] += 1
                    if train_mask.any():
                        loss = self.loss(base_log_probs[train_mask], target_log_probs[train_mask])
                        current_batch_loss += loss
                        losses.append(loss.item())
                last_hero_tokens = hero_tokens
            current_batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            mean_loss = np.mean(losses) if losses else 0
            print(f"Itr {itr} | Mean Loss: {mean_loss:.6f}")
            losses = []
            if itr % 100 == 0 and itr > 0:
                torch.save(self.model.state_dict(), f"RL-{itr}.pt")

if __name__ == '__main__':
    sim = Simulator('checkpoint-140000')
    sim.rl()

