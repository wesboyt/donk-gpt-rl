
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
        batch_size = 512
        for itr in range(0, 5000000):
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
                        vals = np.array([evs[tk] for tk in valid_tokens])
                        max_val = np.max(vals)
                        exp_vals = np.exp(vals - max_val)
                        dist_probs = exp_vals / np.sum(exp_vals)
                        target_probs[idx, t - 1] = 1e-6
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

            if itr % 5 == 0 and itr > 0:
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
