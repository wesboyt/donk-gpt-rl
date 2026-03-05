import sys
import os
import copy
import json
from random import randint
import traceback
import numpy as np
import gc
import csv

sys.modules["markupsafe._speedups"] = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import schedulefree
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from sim_hand import Hand
from sim_encoder import Encoder


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0


    def push(self, data_dict):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data_dict
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class Simulator:
    def __init__(self):
        self.device = 'cuda'
        torch.set_float32_matmul_precision('high')

        config = AutoConfig.from_pretrained('./config.json')

        # Training Model
        self.model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))

        # Reference Model for KL Divergence
        self.ref_model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.ref_model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.unk_token_id = self.tokenizer.unk_token_id

        self.result_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.fold_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.check_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.call_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.raise_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.allin_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)

        self.raise_token_id = self.raise_token.item()
        self.check_token_id = self.check_token.item()
        self.call_token_id = self.call_token.item()
        self.allin_token_id = self.allin_token.item()
        self.fold_token_id = self.fold_token.item()

        self.min_size_token_id = self.tokenizer.encode("<b1%>")[0]
        self.min_size_token = torch.tensor([self.min_size_token_id]).to(self.device)

        # Core Tokens
        self.hero_token_ids = torch.tensor([self.tokenizer.encode(f"<herop{i}>")[0] for i in range(6)], device=self.device)
        self.action_tokens = {
            'fold': self.fold_token_id,
            'check': self.check_token_id,
            'call': self.call_token_id,
            'raise': self.raise_token_id,
            'allin': self.allin_token_id
        }

        # Token-Space Regret Supports
        self.sizes = np.array(list(range(1, 5)) + list(range(5, 101, 5)) + list(range(125, 501, 25)), dtype=np.float32)
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()

        # Optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(), lr=1e-6, warmup_steps=0, betas=(0.9, 0.999), weight_decay=0.01
        )
        self.optimizer.train()

        # Engine Params
        self.n_sims = 16
        self.batch_size = 64
        self.global_updates = 0

        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.encoder = Encoder()

        self.log_file = open('training_logs.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['Update', 'Total_Loss', 'Strategy_Loss', 'Regret_Loss'])

    def generate_experience(self):
        itr_street_cutoffs = [3, 2, 1, 0]
        hand_street_cutoffs = [5, 4, 3, 0]

        itr_street_index = min((self.global_updates // 10000), len(itr_street_cutoffs) - 1)

        hands, hero_ev_data, hero_indices = self.batch_generate_hands(
            self.batch_size, itr_street_cutoffs[itr_street_index]
        )

        for i, hand in enumerate(hands):
            if not hero_ev_data[i]: continue

            hero_idx = hero_indices[i]
            big_blind = float(hand.big_blind) if hand.big_blind > 0 else 1.0

            for node_data in hero_ev_data[i]:
                # Unpack the temporally frozen state and its perfectly aligned EVs
                state_json = node_data['state_json']
                ev_dict = node_data['evs']

                parsed_uh = json.loads(state_json)
                if not parsed_uh[0][hero_idx] or len(parsed_uh[1]) < hand_street_cutoffs[itr_street_index]:
                    continue

                encoded_str = self.encoder.encode(state_json)
                base_encoded = f"{encoded_str}<herop{hero_idx}>"

                # Feed the frozen ev_dict directly into the means calculation
                means = {
                    act_str: float(np.mean(samples)) / big_blind
                    for act_str, samples in ev_dict.items()
                    if act_str in self.action_tokens and samples
                }

                if not means: continue

                acts = list(means.keys())
                scores = np.array([means[a] for a in acts])
                max_score = np.max(scores)

                # Package the immutable environment data
                chosen_act = np.random.choice(acts)
                experience = {
                    'strategy_text': base_encoded,
                    'legal_actions': acts,
                    'ev_scores': scores.tolist(),
                    'regret_text': f"{base_encoded}<{chosen_act}><unk>",
                    'regret_target': max_score - means[chosen_act]
                }

                self.replay_buffer.push(experience)

    def rl(self):
        print("Starting Joint Training: Sequential Generation + Replay Buffer...")
        beta = 1.0

        while self.global_updates < 40000:
            # Sequentially generate a batch of rollouts and push them to disk history
            try:
                self.generate_experience()
            except Exception as e:
                print(f"Generator Exception: {e}")
                traceback.print_exc()

            # Wait until buffer has enough history
            if len(self.replay_buffer) < self.batch_size:
                continue

            # Sample randomly from the entire history
            samples = self.replay_buffer.sample(self.batch_size)

            batch_data = {
                'strategy_text': [s['strategy_text'] for s in samples],
                'legal_actions': [s['legal_actions'] for s in samples],
                'ev_scores': [s['ev_scores'] for s in samples],
                'regret_text': [s['regret_text'] for s in samples],
                'regret_targets': [s['regret_target'] for s in samples]
            }

            # Strategy Forward Pass
            strat_inputs = self.tokenizer(batch_data['strategy_text'], padding=True, return_tensors="pt")
            strat_ids = strat_inputs.input_ids.to(self.device, non_blocking=True)
            strat_mask = strat_inputs.attention_mask.to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                strat_logits = self.model(strat_ids, attention_mask=strat_mask).logits.float()

            s_preds = strat_logits[:, -1, :]

            # Isolate strictly the 5 action logits
            action_ids_list = [self.fold_token_id, self.check_token_id, self.call_token_id, self.raise_token_id, self.allin_token_id]
            action_tensor = torch.tensor(action_ids_list, device=self.device)
            s_preds_actions = s_preds[:, action_tensor]

            s_targets_actions = torch.zeros_like(s_preds_actions)
            illegal_mask = torch.ones_like(s_preds_actions, dtype=torch.bool)
            dynamic_weights = []

            with torch.no_grad():
                for b in range(self.batch_size):
                    acts = batch_data['legal_actions'][b]
                    scores = np.array(batch_data['ev_scores'][b])
                    max_score = np.max(scores)

                    # Map legal actions to our 5-dim tensor
                    local_indices = [action_ids_list.index(self.action_tokens[act]) for act in acts]
                    illegal_mask[b, local_indices] = False

                    # Extract CURRENT model probabilities for calculating v_s
                    valid_logits = s_preds_actions[b, local_indices]
                    valid_probs = F.softmax(valid_logits, dim=-1).cpu().numpy()

                    # Calculate Expected Value of the CURRENT policy
                    v_s = np.sum(valid_probs * scores)

                    # Calculate Instantaneous Regrets dynamically
                    regrets = np.maximum(scores - v_s, 0)
                    regret_sum = np.sum(regrets)

                    if regret_sum > 0:
                        target_probs = regrets / regret_sum
                    else:
                        best_acts = (scores >= max_score - 1e-4)
                        target_probs = best_acts.astype(float) / np.sum(best_acts)

                    for i, idx in enumerate(local_indices):
                        s_targets_actions[b, idx] = target_probs[i]

                    # Baseline weight of 1.0, max weight of ~8.0 for massive river mistakes
                    ev_spread = max_score - np.min(scores)
                    weight = float(ev_spread) + 1e-3  # Tiny floor to prevent strict zero division
                    dynamic_weights.append(weight)

            s_preds_actions = s_preds_actions.masked_fill(illegal_mask, -1e2)

            s_loss_unreduced = F.cross_entropy(s_preds_actions, s_targets_actions, reduction='none')

            # Apply normalized dynamic weights
            batch_weights = torch.tensor(dynamic_weights, device=self.device)
            batch_weights = batch_weights / batch_weights.mean()
            s_loss = torch.mean(s_loss_unreduced * batch_weights)

            total_loss = s_loss
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.global_updates += 1
            if self.global_updates % 10 == 0:
                self.csv_writer.writerow([self.global_updates, total_loss.item(), s_loss.item()])
                self.log_file.flush()

            if self.global_updates % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Upd {self.global_updates} | Total: {total_loss.item():.4f} (Strat: {s_loss.item():.4f})")
                self.model.eval()
                torch.save(self.model.state_dict(), f"RL-{self.global_updates}.pt")
                self.model.train()

    def batch_generate_hands(self, n_hands, street_cutoff):
        hands = [Hand() for _ in range(n_hands)]
        hero_indices = [randint(0, 5) for _ in range(n_hands)]
        hero_ev_data = [[] for _ in range(n_hands)]
        active_indices = list(range(n_hands))

        while active_indices:
            current_hands = [hands[i] for i in active_indices]
            ev_indices = []
            ev_hand_objs = []

            for i, idx in enumerate(active_indices):
                hand = hands[idx]
                if not hand.done and hand.state.turn_index == hero_indices[idx] and hand.state.street_index >= street_cutoff:
                    ev_indices.append(idx)
                    ev_hand_objs.append(hand)

            if ev_hand_objs:
                cf_sizes = self.select_raise_batch(ev_hand_objs)
                bulk_results = self.generate_bulk_evs(ev_hand_objs, cf_sizes)
                for k, global_idx in enumerate(ev_indices):
                    current_state_json = json.dumps(ev_hand_objs[k].get_u_hand(hero_indices[global_idx]))
                    hero_ev_data[global_idx].append({
                        'state_json': current_state_json,
                        'evs': bulk_results[k]
                    })

            actions, sizes = self.select_action_batch(current_hands)
            next_active_indices = []

            for i, (action, size) in enumerate(zip(actions, sizes)):
                hand_idx = active_indices[i]
                hand = hands[hand_idx]

                if action == 'fold':
                    hand.fold()
                elif action == 'check':
                    hand.check()
                elif action == 'call':
                    hand.call()
                elif action == 'raise':
                    action_space = hand.get_action_space()
                    if size >= (0.5 * action_space.get('max_bet', 0)):
                        hand.bet_or_raise(action_space.get('max_bet', 0))
                    else:
                        hand.bet_or_raise(size)
                elif action == 'allin':
                    action_space = hand.get_action_space()
                    if 'max_bet' in action_space:
                        hand.bet_or_raise(action_space['max_bet'])
                    else:
                        hand.call()

                if not hand.done: next_active_indices.append(hand_idx)

            active_indices = next_active_indices

        return hands, hero_ev_data, hero_indices

    @torch.inference_mode()
    def generate_bulk_evs(self, hand_list, size_list):
        all_sims = []
        registry = []

        for i, hand in enumerate(hand_list):
            action_space = hand.get_action_space()
            raise_size = size_list[i]
            player = hand.state.turn_index

            res = {
                'pot_size': hand.pot_size(),
                'call_size': action_space.get('call', hand.big_blind),
                'can_allin': 'max_bet' in action_space
            }

            if 'fold' in action_space:
                res['fold'] = [-hand.investment()] * self.n_sims

            registry.append({'holder': res, 'is_calc': False})

            max_bet = action_space.get('max_bet', 0)
            valid_actions = {'fold', 'check', 'call', 'min_bet', 'max_bet'}

            # Mirror the exact boolean logic from process_hand_cpu
            can_raise = 'min_bet' in action_space
            if can_raise and action_space['min_bet'] >= max_bet:
                can_raise = False

            can_call = 'call' in action_space
            call_is_allin = can_call and not can_raise

            # Discard branches based strictly on the synchronized logic
            if not can_raise:
                valid_actions.discard('min_bet')

            if call_is_allin:
                # If call is a forced all-in, discard max_bet to prevent duplicate tree simulation
                valid_actions.discard('max_bet')
            # ---------------------------------

            for root_action in action_space.keys():
                if root_action not in valid_actions or root_action == 'fold':
                    continue

                temp_hand = copy.deepcopy(hand)
                if root_action == 'check':
                    temp_hand.check()
                elif root_action == 'call':
                    temp_hand.call()
                elif root_action == 'min_bet':
                    temp_hand.bet_or_raise(raise_size)
                elif root_action == 'max_bet':
                    temp_hand.bet_or_raise(max_bet)

                if temp_hand.done:
                    p = temp_hand.state.payoffs
                    for j in range(len(p)):
                        if p[j] > 0: p[j] -= min(p[j] * 0.05, 2 * temp_hand.big_blind)
                    res[root_action] = [p[player]] * self.n_sims
                else:
                    start_idx = len(all_sims)
                    for _ in range(self.n_sims):
                        sim_clone = copy.deepcopy(temp_hand)
                        sim_clone.shuffle()
                        all_sims.append(sim_clone)
                    registry.append({'start': start_idx, 'count': self.n_sims, 'player': player, 'action': root_action, 'is_calc': True})
                    res[root_action] = "PENDING"

        active_sim_indices = list(range(len(all_sims)))
        finished_payoffs = [None] * len(all_sims)

        while active_sim_indices:
            current_sim_batch = [all_sims[k] for k in active_sim_indices]
            actions, sizes = self.select_action_batch(current_sim_batch)
            next_active = []

            for j, (act, sz) in enumerate(zip(actions, sizes)):
                sim_idx = active_sim_indices[j]
                sim = all_sims[sim_idx]

                if act == 'fold':
                    sim.fold()
                elif act == 'check':
                    sim.check()
                elif act == 'call':
                    sim.call()
                elif act == 'allin':
                    action_space = sim.get_action_space()
                    if 'max_bet' in action_space:
                        sim.bet_or_raise(action_space['max_bet'])
                    else:
                        sim.call()
                elif act == 'raise':
                    action_space = sim.get_action_space()
                    # Keep the threshold here ONLY for environment execution
                    if sz >= (0.5 * action_space.get('max_bet', 0)):
                        sim.bet_or_raise(action_space.get('max_bet', 0))
                    else:
                        sim.bet_or_raise(sz)

                if sim.done:
                    p = sim.state.payoffs
                    for x in range(len(p)):
                        if p[x] > 0: p[x] -= min(p[x] * 0.05, 2 * sim.big_blind)
                    finished_payoffs[sim_idx] = p
                else:
                    next_active.append(sim_idx)
            active_sim_indices = next_active

        final_output = []
        current_holder = None

        for item in registry:
            if not item['is_calc']:
                if current_holder is not None: final_output.append(current_holder)
                current_holder = item['holder']
            else:
                start, count, plyr, act = item['start'], item['count'], item['player'], item['action']
                current_holder[act] = [finished_payoffs[k][plyr] for k in range(start, start + count)]

        if current_holder is not None:
            final_output.append(current_holder)

        for i, hand in enumerate(hand_list):
            action_space = hand.get_action_space()
            max_bet = action_space.get('max_bet', 0)

            # Re-evaluate the booleans to ensure final token mapping is completely gapless
            can_raise = 'min_bet' in action_space
            if can_raise and action_space['min_bet'] >= max_bet:
                can_raise = False

            can_call = 'call' in action_space
            call_is_allin = can_call and not can_raise

            if 'min_bet' in final_output[i]:
                final_output[i]['raise'] = final_output[i].pop('min_bet')
            if 'max_bet' in final_output[i]:
                final_output[i]['allin'] = final_output[i].pop('max_bet')

            # If the inference mask forces a call to be an allin, the target must also be an allin
            if call_is_allin and 'call' in final_output[i]:
                final_output[i]['allin'] = final_output[i].pop('call')
            # ---------------------------------

        return final_output


    @torch.inference_mode()
    def select_raise_batch(self, hands):
        if not hands: return []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, _, _, _, _, actor_indices = zip(*results)

        inputs = self.tokenizer(list(encoded_strs), padding=True, return_tensors="pt")

        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
        batch_actor_tokens = self.hero_token_ids[list(actor_indices)].to(self.device)
        raise_col = torch.full((len(hands), 1), self.raise_token_id, device=self.device, dtype=torch.long)

        final_input_ids = torch.cat([input_ids, batch_actor_tokens.unsqueeze(1), raise_col], dim=1)
        extension_mask = torch.ones((len(hands), 2), device=self.device, dtype=torch.long)
        final_attention_mask = torch.cat([attention_mask, extension_mask], dim=1)

        logits = self.model(final_input_ids, attention_mask=final_attention_mask).logits[:, -1, :]

        start_id = self.min_size_token.item()
        num_sizes = len(self.sizes_floats)
        size_logits = logits[:, start_id: start_id + num_sizes]

        batch_min_tokens = torch.tensor(min_bet_tokens, device=self.device)
        offsets = (batch_min_tokens - start_id).unsqueeze(1)
        size_indices = torch.arange(num_sizes, device=self.device).unsqueeze(0)

        size_logits = size_logits.masked_fill(~(size_indices >= offsets), float('-inf'))
        relative_indices = torch.multinomial(torch.softmax(size_logits, dim=1), num_samples=1).squeeze(1)
        chosen_percents = self.torch_sizes_float[relative_indices]

        bets = (torch.tensor(pot_sizes, device=self.device) * (chosen_percents / 100.0)).long()

        final_bets = [
            int(min(amt, cap) if cap > 0 else amt)
            for amt, cap in zip(bets.tolist(), max_bets)
        ]

        del inputs, input_ids, attention_mask, batch_actor_tokens, raise_col
        del final_input_ids, extension_mask, final_attention_mask, logits
        del size_logits, batch_min_tokens, offsets, size_indices, relative_indices
        del chosen_percents, bets

        return final_bets

    @torch.inference_mode()
    def select_action_batch(self, hands):
        if not hands: return [], []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices = zip(*results)

        queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}>" for i in range(len(hands))]

        inputs = self.tokenizer(queries, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        logits = self.model(input_ids, attention_mask=attention_mask).logits[:, -1, :]

        action_keys = ['fold', 'check', 'call', 'raise', 'allin']
        action_ids = [self.fold_token_id, self.check_token_id, self.call_token_id, self.raise_token_id, self.allin_token_id]
        action_tensor = torch.tensor(action_ids, device=self.device)

        action_logits = logits[:, action_tensor]

        final_actions = [''] * len(hands)
        final_sizes = [0] * len(hands)
        raise_indexes = []

        for i in range(len(hands)):
            legal_mask = torch.zeros(5, dtype=torch.bool, device=self.device)

            if can_check[i]:
                legal_mask[1] = True
            else:
                legal_mask[0] = True

            if can_call[i] and not call_is_allin[i]: legal_mask[2] = True
            if can_raise[i]: legal_mask[3] = True
            if max_bets[i] > 0 or call_is_allin[i]: legal_mask[4] = True

            #action_logits[i].masked_fill_(~legal_mask, float('-inf'))

            # DO NOT mask the logits with -inf
        action_probs = F.softmax(action_logits, dim=-1)

        # Apply the legal mask directly to the probabilities
        legal_mask_tensor = torch.stack([
            torch.tensor([
                not can_check[i],  # fold
                can_check[i],  # check
                can_call[i] and not call_is_allin[i],  # call
                can_raise[i],  # raise
                max_bets[i] > 0 or call_is_allin[i]  # allin
            ], device=self.device) for i in range(len(hands))
        ])

        # Zero out illegal probabilities
        action_probs = action_probs * legal_mask_tensor.float()

        # Re-normalize so they sum back to 1.0 safely
        sum_probs = action_probs.sum(dim=-1, keepdim=True)
        action_probs = torch.where(sum_probs > 0, action_probs / sum_probs, legal_mask_tensor.float() / legal_mask_tensor.sum(dim=-1, keepdim=True))

        chosen_indices = torch.multinomial(action_probs, num_samples=1).squeeze(1)

        for i in range(len(hands)):
            chosen_act = action_keys[chosen_indices[i].item()]
            final_actions[i] = chosen_act

            if chosen_act == 'allin':
                final_sizes[i] = max_bets[i]
            elif chosen_act == 'raise':
                raise_indexes.append(i)

        del inputs, input_ids, attention_mask, logits, action_logits, action_probs

        if raise_indexes:
            start_id = self.min_size_token.item()
            num_sizes = len(self.sizes_floats)

            r_encoded = [encoded_strs[i] for i in raise_indexes]
            r_inputs = self.tokenizer(r_encoded, padding=True, return_tensors="pt")
            r_ids = r_inputs.input_ids.to(self.device, non_blocking=True)
            r_mask = r_inputs.attention_mask.to(self.device, non_blocking=True)
            r_actors = self.hero_token_ids[list(np.array(actor_indices)[raise_indexes])].to(self.device)
            r_raise_tok = torch.full((len(raise_indexes), 1), self.raise_token_id, device=self.device)

            r_final_ids = torch.cat([r_ids, r_actors.unsqueeze(1), r_raise_tok], dim=1)
            r_final_mask = torch.cat([r_mask, torch.ones((len(raise_indexes), 2), device=self.device)], dim=1)

            r_logits = self.model(r_final_ids, attention_mask=r_final_mask).logits[:, -1, :]
            size_logits = r_logits[:, start_id: start_id + num_sizes]

            r_min_tokens = torch.tensor([min_bet_tokens[i] for i in raise_indexes], device=self.device)
            offsets = (r_min_tokens - start_id).unsqueeze(1)
            size_indices = torch.arange(num_sizes, device=self.device).unsqueeze(0)

            size_logits = size_logits.masked_fill(~(size_indices >= offsets), float('-inf'))
            r_indices = torch.multinomial(torch.softmax(size_logits, dim=1), num_samples=1).squeeze(1)
            r_chosen_pct = self.torch_sizes_float[r_indices]

            r_bets = (torch.tensor([pot_sizes[i] for i in raise_indexes], device=self.device) * (
                    r_chosen_pct / 100.0)).long().tolist()

            for k, hand_idx in enumerate(raise_indexes):
                final_sizes[hand_idx] = int(min(r_bets[k], max_bets[hand_idx]))

            del r_inputs, r_ids, r_mask, r_actors, r_raise_tok, r_final_ids, r_final_mask
            del r_logits, size_logits, r_min_tokens, offsets, size_indices, r_indices, r_chosen_pct

        return final_actions, final_sizes

    def process_hand_cpu(self, hand):
        action_space = hand.get_action_space()
        pot_size = hand.pot_size()

        if 'min_bet' in action_space:
            idx = np.searchsorted(self.sizes_floats, (action_space['min_bet'] / pot_size) * 100, side='left')
            min_bet_token = self.min_size_token_id + min(idx, len(self.sizes_floats) - 1)
            max_bet = action_space['max_bet']
        else:
            min_bet_token = 0
            max_bet = 0

        turn_idx = hand.state.turn_index

        can_check = 'check' in action_space
        can_raise = 'min_bet' in action_space
        can_call = 'call' in action_space

        if can_raise and action_space['min_bet'] >= max_bet:
            can_raise = False

        call_is_allin = can_call and not can_raise

        return (
            self.encoder.encode(json.dumps(hand.get_u_hand(turn_idx))),
            min_bet_token,
            max_bet,
            pot_size,
            can_check,
            can_raise,
            can_call,
            call_is_allin,
            turn_idx
        )


if __name__ == '__main__':
    sim = Simulator()
    sim.rl()
