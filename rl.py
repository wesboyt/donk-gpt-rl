
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
from random import random, randint
import schedulefree
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, AutoConfig
import json
from sim_hand import Hand
import torch
import numpy as np
from sim_encoder import Encoder
import gc
import pickle


class Simulator:
    def __init__(self):
        self.device = 'cuda'
        config = AutoConfig.from_pretrained('./config.json')
        self.model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.ref_model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.ref_model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.tokenizer = GPT2TokenizerFast.from_pretrained('./opt-it-2')
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.result_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.fold_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.flop_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.check_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.call_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.raise_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.allin_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.min_size_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.sizes = list(range(1, 5))
        self.sizes = np.int16(self.sizes)
        self.torch_sizes = torch.tensor(self.sizes).to(self.device)
        self.torch_sizes_float = self.torch_sizes.float()
        self.encoder = Encoder()
        self.loss = torch.nn.KLDivLoss(reduction="sum", log_target=True).to(self.device)
        self.zero_tensor = torch.tensor(0.0, device=self.device)
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=3e-4)
        self.optimizer.train()

    def rl(self):
        losses = []
        batch_size = 256

        for itr in range(0, 50000):
            batch_text = []
            batch_hero_evs = []
            for i in range(batch_size):
                player = randint(0, 5)
                hero_ev_list = []

                with torch.no_grad():
                    hand = Hand()
                    while not hand.done:
                        if hand.state.turn_index == player:
                            evs = self.generate_action_evs(hand)
                            hero_ev_list.append(evs)
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

                    uh = hand.get_u_hand(player)
                    batch_text.append(self.encoder.encode(json.dumps(uh), True))
                    batch_hero_evs.append(hero_ev_list)
            inputs = self.tokenizer(batch_text, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            seq_len = input_ids.shape[1]
            hero_ids = input_ids[:, 1]
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
                target_probs = torch.softmax(ref_outputs.logits, dim=2)
            ev_indexes = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            start_index = 26

            for t in range(start_index, seq_len):
                current_tokens = input_ids[:, t]
                prev_tokens = input_ids[:, t - 1]
                is_action_token = ((current_tokens >= 9) & (current_tokens <= 13))
                is_hero_turn = (prev_tokens == hero_ids)
                target_mask = is_action_token & is_hero_turn

                if target_mask.any():
                    idxs = torch.nonzero(target_mask).squeeze()
                    if idxs.ndim == 0:
                        idxs = idxs.unsqueeze(0)

                    for idx in idxs:
                        idx = idx.item()
                        evs = batch_hero_evs[idx][ev_indexes[idx].item()]
                        ev_indexes[idx] += 1
                        valid_tokens = list(evs.keys())
                        vals = np.array([evs[tk] for tk in valid_tokens]) / .5
                        max_val = np.max(vals)
                        exp_vals = np.exp(vals - max_val)
                        dist_probs = exp_vals / np.sum(exp_vals)
                        target_probs[idx, t - 1] = 0
                        for j, tk in enumerate(valid_tokens):
                            target_probs[idx, t - 1, tk] = max(float(dist_probs[j]), 1e-6)
                        target_probs[idx, t - 1] /= target_probs[idx, t - 1].sum()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_target_probs = target_probs[:, :-1, :].contiguous()
            final_mask = attention_mask[:, 1:].contiguous()

            loss = torch.nn.functional.kl_div(
                torch.log_softmax(shift_logits, dim=2),
                torch.log(shift_target_probs + 1e-9),
                reduction='none',
                log_target=True
            ).sum(dim=2)

            final_loss = (loss * final_mask).sum() / final_mask.sum()
            losses.append(final_loss.item())
            final_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if itr % 15 == 0 and itr > 0:
                print(f"Itr {itr} | Loss: {np.mean(losses):.6f}")
                losses = []
                self.model.eval()
                torch.save(self.model.state_dict(), f"RL-{itr}.pt")
                self.model.train()


    @torch.inference_mode()
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
                target[self.fold_token] = chance
            if 'check' in action_space:
                chance = likelihoods[self.check_token]
                target[self.check_token] = chance
            if 'call' in action_space:
                chance = likelihoods[self.call_token]
                target[self.call_token] = chance
            if 'min_bet' in action_space:
                chance = likelihoods[self.raise_token]
                target[self.raise_token] = chance
            effective_stack = hnd.state.get_effective_stack(hnd.state.turn_index)
            if ('call' in action_space and effective_stack == hnd.state.checking_or_calling_amount) or 'min_bet' in action_space and action_space['min_bet'] >= .25 * action_space['max_bet']:
                chance = likelihoods[self.allin_token]
                target[self.allin_token] = chance
            target = target / target.sum()
            return target

    @torch.inference_mode()
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

    def generate_payoffs(self, ohand):
        with torch.no_grad():
            payoffs = []
            for i in range(32):
                hand = copy.deepcopy(ohand)
                hand.shuffle()
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
                for j, val in enumerate(payoff):
                    if payoff[j] > 0:
                        tax =  min(payoff[j] * .05, 2 * hand.big_blind)
                        payoff[j] -= tax
                payoffs.append(payoff)
            return payoffs

    @torch.inference_mode()
    def select_action_batch(self, hands):
        with torch.no_grad():
            batch_size = len(hands)
            if batch_size == 0: 
                return []
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
                if 'call' in action_space: 
                    valid_probs['call'] = call_probs[i].item()
                if 'min_bet' in action_space:
                    valid_probs['raise'] = raise_probs[i].item() + allin_probs[i].item()
                elif 'call' in action_space:
                    valid_probs['call'] += allin_probs[i].item()  # Add allin prob to call if raise illegal

                if 'fold' in action_space: 
                    valid_probs['fold'] = fold_probs[i].item()
                if 'check' in action_space: 
                    valid_probs['check'] = check_probs[i].item()

                total_p = sum(valid_probs.values())
                if total_p == 0: 
                    total_p = 1e-9

                if 'raise' in valid_probs:
                    prob = valid_probs['raise'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        actions[i] = ('raise', 0)
                        raise_indices.append(i)
                        selected = True

                if not selected and 'call' in valid_probs:
                    prob = valid_probs['call'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        actions[i] = ('call', hnd.state.checking_or_calling_amount)
                        selected = True

                if not selected and 'fold' in valid_probs:
                    prob = valid_probs['fold'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        actions[i] = ('fold', 0)
                        selected = True

                if not selected and 'check' in valid_probs:
                    prob = valid_probs['check'] / total_p
                    cumulative += prob
                    if cumulative >= roll:
                        actions[i] = ('check', 0)
                        selected = True
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

    @torch.inference_mode()
    def select_raise_batch(self, hands):
        if not hands: 
            return []
        batch_size = len(hands)

        with torch.no_grad():
            batch_strs = [self.encoder.encode(json.dumps(h.get_u_hand())) for h in hands]
            input_ids = self.tokenizer(batch_strs, padding=True, return_tensors='pt').input_ids.to(self.device)
            raise_col = torch.full((batch_size, 1), self.raise_token.item(), device=self.device)
            input_ids = torch.cat([input_ids, raise_col], dim=1)
            logits = self.model(input_ids).logits[:, -1, :]
            start_idx = self.min_size_token.item()
            end_idx = start_idx + len(self.sizes)
            size_logits = logits[:, start_idx:end_idx]
            all_probs = torch.softmax(size_logits, dim=1).cpu().numpy()
        results = []
        for i, hnd in enumerate(hands):
            action_space = hnd.get_action_space()
            pot_size = hnd.pot_size()
            min_bet = action_space.get('min_bet', 0)
            valid_start_index = 0
            for idx, size_pct in enumerate(self.sizes):
                if pot_size * (size_pct / 100) >= min_bet:
                    valid_start_index = idx
                    break
            hand_probs = all_probs[i, valid_start_index:]
            prob_sum = hand_probs.sum()
            if prob_sum == 0:
                hand_probs = np.ones_like(hand_probs) / len(hand_probs)
            else:
                hand_probs /= prob_sum
            roll = random()
            cumulative = 0
            chosen_offset = 0
            for idx, p in enumerate(hand_probs):
                cumulative += p
                if cumulative >= roll:
                    chosen_offset = idx
                    break
            final_idx = valid_start_index + chosen_offset
            pct = self.sizes[min(final_idx, len(self.sizes) - 1)]
            bet = int(pot_size * (pct / 100))
            if 'max_bet' in action_space:
                if bet >= 0.5 * action_space['max_bet']:
                    bet = action_space['max_bet']
                elif bet > action_space['max_bet']:
                    bet = action_space['max_bet']

            results.append(bet)

        return results

    @torch.inference_mode()
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

    @torch.inference_mode()
    def generate_action_evs(self, ohand):
        player = ohand.state.turn_index
        action_space = ohand.get_action_space()
        scenarios = []
        n_sims = 32
        tokens_map = {}
        raise_hands_to_process = []
        raise_indices_in_scenarios = []

        def add_scenario(action_type, target_list=None):
            h = copy.deepcopy(ohand)
            h.shuffle()

            if action_type == 'fold':
                h.fold()
            elif action_type == 'check':
                h.check()
            elif action_type == 'call':
                h.call()
            elif action_type == 'raise':
                # Defer the actual action until we have the size
                pass

            scenarios.append(h)
            if target_list is not None:
                target_list.append(h)
                raise_indices_in_scenarios.append(len(scenarios) - 1)
        if 'fold' in action_space:
            start = len(scenarios)
            for _ in range(n_sims): 
                add_scenario('fold')
            tokens_map[self.fold_token.item()] = list(range(start, len(scenarios)))

        if 'check' in action_space:
            start = len(scenarios)
            for _ in range(n_sims): 
                add_scenario('check')
            tokens_map[self.check_token.item()] = list(range(start, len(scenarios)))

        if 'call' in action_space:
            start = len(scenarios)
            for _ in range(n_sims): 
                add_scenario('call')
            tokens_map[self.call_token.item()] = list(range(start, len(scenarios)))

        if 'min_bet' in action_space:
            start = len(scenarios)
            for _ in range(n_sims): 
                add_scenario('raise', raise_hands_to_process)
            tokens_map[self.raise_token.item()] = list(range(start, len(scenarios)))
        if raise_hands_to_process:
            sizes = self.select_raise_batch(raise_hands_to_process)
            for h, size in zip(raise_hands_to_process, sizes):
                h.bet_or_raise(size)
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
                if p > 0:
                    tax = min(p * .05, 2 * h.big_blind)
                    p -= tax
                payoff_sum += p
                count += 1
            if count > 0:
                results[token] = payoff_sum / count

        return results

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
                
    @torch.inference_mode()
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
    sim = Simulator()
    sim.rl()
