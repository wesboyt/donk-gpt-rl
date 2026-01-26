import copy
from random import random, randint
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2TokenizerFast, AutoConfig
import schedulefree
from sim_hand import Hand
from sim_encoder import Encoder

#this is a file that compares two poker models and proves with confidence intervals which is stronger.

class Simulator:
    def __init__(self, config_path_a: str, weights_path_a: str, config_path_b: str, weights_path_b: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_a = AutoConfig.from_pretrained(config_path_a)
        self.model_a = AutoModelForCausalLM.from_config(config_a)
        self.model_a.load_state_dict(torch.load(weights_path_a, map_location=self.device))
        self.model_a.to(self.device)
        self.model_a.eval()
        config_b = AutoConfig.from_pretrained(config_path_b)
        self.model_b = AutoModelForCausalLM.from_config(config_b)
        self.model_b.load_state_dict(torch.load(weights_path_b, map_location=self.device))
        self.model_b.to(self.device)
        self.model_b.eval()
        self.tokenizer = GPT2TokenizerFast.from_pretrained('./opt-it-2')
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.result_token = torch.tensor(self.tokenizer.encode("<result>")).to(self.device)
        self.fold_token = torch.tensor(self.tokenizer.encode("<fold>")).to(self.device)
        self.check_token = torch.tensor(self.tokenizer.encode("<check>")).to(self.device)
        self.call_token = torch.tensor(self.tokenizer.encode("<call>")).to(self.device)
        self.raise_token = torch.tensor(self.tokenizer.encode("<raise>")).to(self.device)
        self.allin_token = torch.tensor(self.tokenizer.encode("<allin>")).to(self.device)
        self.min_size_token = torch.tensor(self.tokenizer.encode("<b1%>")).to(self.device)
        self.sizes = list(range(1, 5))
        self.sizes.extend(list(range(5, 101, 5)))
        self.sizes.extend(list(range(125, 501, 25)))
        self.sizes = np.int16(self.sizes)
        self.torch_sizes = torch.tensor(self.sizes).to(self.device)
        self.torch_sizes_float = self.torch_sizes.float()
        self.encoder = Encoder()


    def compare_strategies(self, gamestate, n_sims=100):
        """
        Main entry point.
        Takes a gamestate and compares Model A vs Model B.
        Returns: tuple(EV_Model_A, EV_Model_B)
        """
        hero_index = gamestate.state.turn_index
        ev_a = self._simulate_rollout_ev(
            gamestate,
            hero_model=self.model_a,
            villain_model=self.model_b,
            hero_index=hero_index,
            n_sims=n_sims
        )
        ev_b = self._simulate_rollout_ev(
            gamestate,
            hero_model=self.model_b,
            villain_model=self.model_a,
            hero_index=hero_index,
            n_sims=n_sims
        )

        return (ev_a, ev_b)

    def _simulate_rollout_ev(self, gamestate, hero_model, villain_model, hero_index, n_sims):
        """
        Runs Monte Carlo simulations for a specific matchup configuration.
        """
        total_payoff = 0.0

        for _ in range(n_sims):
            hand = copy.deepcopy(gamestate)
            hand.shuffle()

            while not hand.done:
                current_turn = hand.state.turn_index
                if current_turn == hero_index:
                    active_model = hero_model
                else:
                    active_model = villain_model
                action, size = self.select_action(hand, model=active_model)

                match action:
                    case 'fold':
                        hand.fold()
                    case 'check':
                        hand.check()
                    case 'call':
                        hand.call()
                    case 'raise':
                        hand.bet_or_raise(size)
            total_payoff += hand.state.payoffs[hero_index]
        return total_payoff / n_sims

    def select_action(self, hnd, model):
        with torch.no_grad():
            action_space = hnd.get_action_space()
            target = self.get_action_likelihoods(hnd, model)
            roll = random()
            count = 0
            if 'min_bet' in action_space:
                count += target[self.raise_token].item()
                if count >= roll:
                    bet = self.select_raise(hnd, model)
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

    def get_action_likelihoods(self, hnd, model):
        with torch.no_grad():
            action_space = hnd.get_action_space()
            uh = hnd.get_u_hand()
            encoded = self.tokenizer.encode(self.encoder.encode(json.dumps(uh)))
            input_ids = torch.tensor(encoded).to(self.device)
            input_ids = torch.cat((input_ids, input_ids[1].unsqueeze(0)), 0).unsqueeze(0)
            logits = model(input_ids).logits[:, -1, :][0]
            likelihoods = torch.softmax(logits, 0)
            target = torch.zeros(likelihoods.shape).to(self.device)

            if 'fold' in action_space:
                target[self.fold_token] = likelihoods[self.fold_token]
            if 'check' in action_space:
                target[self.check_token] = likelihoods[self.check_token]
            if 'call' in action_space:
                target[self.call_token] = likelihoods[self.call_token]
            if 'min_bet' in action_space:
                target[self.raise_token] = likelihoods[self.raise_token]

            effective_stack = hnd.state.get_effective_stack(hnd.state.turn_index)
            if ('call' in action_space and effective_stack == hnd.state.checking_or_calling_amount) or \
                    ('min_bet' in action_space and action_space['min_bet'] >= .25 * action_space['max_bet']):
                target[self.allin_token] = likelihoods[self.allin_token]
            if target.sum() == 0:
                return torch.ones_like(target) / target.shape[0]
            target = target / target.sum()
            return target

    def select_raise(self, hand, model):
        with torch.no_grad():
            bets, min_bet_token = self.get_raise_likelihoods(hand, model)
            roll = random()
            index = 0
            count = bets[index]
            while count < roll and index < bets.shape[0] - 1:
                index += 1
                count += bets[index]

            pot_size = hand.pot_size()
            size_idx = min(int(min_bet_token + index - self.min_size_token), len(self.torch_sizes_float) - 1)
            bet = pot_size * self.torch_sizes_float[size_idx] / 100
            return int(bet)

    def get_raise_likelihoods(self, hnd, model):
        with torch.no_grad():
            action_space = hnd.get_action_space()
            uh = hnd.get_u_hand()
            encoded = self.tokenizer.encode(self.encoder.encode(json.dumps(uh)))
            input_ids = torch.tensor(encoded).to(self.device)
            input_ids = torch.cat((input_ids, input_ids[1].unsqueeze(0), self.raise_token), 0).unsqueeze(0)
            logits = model(input_ids).logits[:, -1, :][0]
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
            if bets.sum() == 0:
                bets = torch.ones_like(bets)
            bets = bets / bets.sum()
            return bets, min_bet_token

if __name__ == '__main__':
    sim = Simulator('config.json', 'GEN-17600000.pt', 'config.json','RL-10000.pt')
    results = []
    for i in range(5000):
        hand = Hand()
        for j in range(4):
            hand.fold()
        result = sim.compare_strategies(hand,100)
        results.append(result[0] - result[1])
    print(np.mean(results), np.std(results))
