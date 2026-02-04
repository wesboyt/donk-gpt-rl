
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
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        try:
            state_dict = torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.ref_model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Warning: Checkpoint not found, initializing random weights.")
        self.tokenizer = GPT2TokenizerFast.from_pretrained('./opt-it-2')
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.result_token = torch.tensor(self.tokenizer.encode("<result>")).to(self.device)
        self.fold_token = torch.tensor(self.tokenizer.encode("<fold>")).to(self.device)
        self.flop_token = torch.tensor(self.tokenizer.encode("<flop>")).to(self.device)
        self.check_token = torch.tensor(self.tokenizer.encode("<check>")).to(self.device)
        self.call_token = torch.tensor(self.tokenizer.encode("<call>")).to(self.device)
        self.raise_token = torch.tensor(self.tokenizer.encode("<raise>")).to(self.device)
        self.allin_token = torch.tensor(self.tokenizer.encode("<allin>")).to(self.device)
        self.min_size_token = torch.tensor(self.tokenizer.encode("<b1%>")).to(self.device)
        self.sizes = list(range(1, 5)) + list(range(5, 101, 5)) + list(range(125, 501, 25))
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
        base_temperature = 0.5
        batch_size = 64
        max_length = 128
        for itr in range(0, 5000000):
            batch_text = []
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
                        states.append(hand_copy)
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
                    batch_states.append(states)
                    batch_text.append(self.encoder.encode(json.dumps(uh), True))
            inputs = self.tokenizer(batch_text, padding='max_length', max_length=max_length, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            hero_ids = input_ids[:, 1]
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits
                target_probs = torch.softmax(ref_logits, dim=2)
            state_indexes = torch.zeros(batch_size, dtype=torch.int).to(self.device)
            last_hero_tokens = torch.tensor([], device=self.device)
            for t in range(1, max_length):
                current_tokens = input_ids[:, t]
                hero_tokens = torch.argwhere(hero_ids == current_tokens).squeeze()
                player_tokens_mask = (current_tokens <= 28) & (current_tokens >= 17)
                state_indexes[player_tokens_mask] += 1
                action_token_mask = (current_tokens <= 13) & (current_tokens >= 9)
                action_indices = torch.argwhere(action_token_mask).squeeze()
                hero_action_indices = torch.tensor([], dtype=torch.long).to(self.device)
                if last_hero_tokens.numel() > 0 and action_indices.numel() > 0:
                    if action_indices.ndim == 0: action_indices = action_indices.unsqueeze(0)
                    if last_hero_tokens.ndim == 0: last_hero_tokens = last_hero_tokens.unsqueeze(0)
                    mask = torch.isin(action_indices, last_hero_tokens)
                    hero_action_indices = action_indices[mask]
                if hero_action_indices.numel() > 0:
                    for idx in hero_action_indices:
                        state_idx = state_indexes[idx].item()
                        if state_idx >= len(batch_states[idx]): continue
                        hero_state = batch_states[idx][state_idx]
                        with torch.no_grad():
                            evs = self.generate_action_evs(hero_state)
                        valid_tokens = list(evs.keys())
                        if not valid_tokens: continue
                        vals = np.array([evs[token] for token in valid_tokens])
                        temp = base_temperature * max(hero_state.big_blind, 1.0)
                        max_val = np.max(vals)
                        exp_vals = np.exp((vals - max_val) / temp)
                        dist_probs = exp_vals / np.sum(exp_vals)
                        for token in valid_tokens:
                            target_probs[idx, t, token] = 0.0
                        for j, token in enumerate(valid_tokens):
                            target_probs[idx, t, token] = max(float(dist_probs[j]), 1e-6)
                        target_probs[idx, t] = target_probs[idx, t] / target_probs[idx, t].sum()

                last_hero_tokens = torch.argwhere(hero_ids == current_tokens).squeeze()

            shift_logits = logits[:, :-1, :].contiguous()
            shift_target_probs = target_probs[:, 1:, :].contiguous()
            shift_attention_mask = attention_mask[:, 1:].contiguous()
            shift_log_probs = torch.log_softmax(shift_logits, dim=2)
            shift_target_log_probs = torch.log(shift_target_probs + 1e-9)
            raw_loss = torch.nn.functional.kl_div(
                shift_log_probs,
                shift_target_log_probs,
                reduction='none',
                log_target=True
            ).sum(dim=2)
            
            masked_loss = raw_loss * shift_attention_mask
            final_loss = masked_loss.sum() / shift_attention_mask.sum()

            final_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(final_loss.item())

            if itr % 100 == 0 and itr > 0:
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
            for i in range(3):
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
                payoffs = self.generate_payoffs(pre_hand)
                results[option_token] = np.mean(list(map(lambda x: x[player], payoffs)))
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
