import sys
import os
import traceback

# -----------------------------------------------------------------------------
# CRITICAL PRE-IMPORT FIXES
# -----------------------------------------------------------------------------
sys.modules["markupsafe._speedups"] = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import queue
import threading
import gc
import concurrent.futures
from random import random, randint
import torch
import schedulefree
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import Counter

from sim_hand import Hand
from sim_encoder import Encoder

class RegretMatchingLoss(torch.nn.Module):
    def __init__(self, ce_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight

    def forward(self, pred_logits, target_probs, action_evs, valid_mask):
        pred_probs = torch.softmax(pred_logits, dim=-1)
        pred_ev = torch.sum(pred_probs * action_evs * valid_mask, dim=-1)
        target_ev = torch.sum(target_probs * action_evs * valid_mask, dim=-1)
        regret_loss = torch.mean(target_ev - pred_ev)
        log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
        ce_loss = -(target_probs * log_probs * valid_mask).sum(dim=-1).mean()

        return regret_loss + (self.ce_weight * ce_loss)

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

        self.model_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.unk_token

        hero_ids = []
        for i in range(6):
            enc = self.tokenizer.encode(f"<herop{i}>")
            hero_ids.append(enc[0])
        self.hero_token_ids = torch.tensor(hero_ids, dtype=torch.long, device=self.device)

        self.result_token = torch.tensor(self.tokenizer.encode("<result>")).to(self.device)
        self.fold_token = torch.tensor(self.tokenizer.encode("<fold>")).to(self.device)
        self.check_token = torch.tensor(self.tokenizer.encode("<check>")).to(self.device)
        self.call_token = torch.tensor(self.tokenizer.encode("<call>")).to(self.device)
        self.raise_token = torch.tensor(self.tokenizer.encode("<raise>")).to(self.device)
        self.raise_token_id = self.raise_token.item()
        self.allin_token = torch.tensor(self.tokenizer.encode("<allin>")).to(self.device)
        self.min_size_token = torch.tensor(self.tokenizer.encode("<b1%>")).to(self.device)
        self.preflop_token = torch.tensor(self.tokenizer.encode("<preflop>")).to(self.device)
        self.flop_token = torch.tensor(self.tokenizer.encode("<flop>")).to(self.device)
        self.turn_token = torch.tensor(self.tokenizer.encode("<turn>")).to(self.device)
        self.river_token = torch.tensor(self.tokenizer.encode("<river>")).to(self.device)

        self.sizes = list(range(1, 5)) + list(range(5, 101, 5)) + list(range(125, 501, 25))
        self.sizes = np.int16(self.sizes)
        self.torch_sizes = torch.tensor(self.sizes).to(self.device)
        self.torch_sizes_float = self.torch_sizes.float()
        self.sizes_floats = np.array(self.torch_sizes_float.tolist(), dtype=np.float32)

        self.regret_loss = RegretMatchingLoss(ce_weight=0.5)

        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=1e-5)
        self.optimizer.train()

        self.n_sims = 16
        self.batch_size = 256
        self.num_generators = 4

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.thread_local = threading.local()

        self.batch_queue = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.global_updates = 0

    def get_thread_tokenizer(self):
        if not hasattr(self.thread_local, 'tokenizer'):
            self.thread_local.tokenizer = copy.deepcopy(self.tokenizer)
        return self.thread_local.tokenizer

    def get_thread_encoders(self, count):
        if not hasattr(self.thread_local, 'encoders'):
            self.thread_local.encoders = []
        while len(self.thread_local.encoders) < count:
            self.thread_local.encoders.append(Encoder())
        return self.thread_local.encoders[:count]

    def get_single_thread_encoder(self):
        if not hasattr(self.thread_local, 'single_encoder'):
            self.thread_local.single_encoder = Encoder()
        return self.thread_local.single_encoder

    def data_generator_worker(self, worker_id):
        itr_street_cutoffs = [3, 2, 1, 0]
        hand_street_cutoffs = [5, 4, 3, 0]

        buffer_text = []
        buffer_evs = []

        while not self.stop_event.is_set():
            updates = self.global_updates
            itr_street_index = 0
            if updates > 0:
                idx = (updates // 20000)
                itr_street_index = min(idx, len(itr_street_cutoffs) - 1)

            while len(buffer_text) < self.batch_size and not self.stop_event.is_set():
                try:
                    hands, hero_ev_data, hero_indices = self.batch_generate_hands(
                        self.batch_size * 2,
                        itr_street_cutoffs[itr_street_index]
                    )

                    local_enc = self.get_single_thread_encoder()

                    for i, hand in enumerate(hands):
                        if len(hero_ev_data[i]) > 0:
                            hero_idx = hero_indices[i]
                            uh = hand.get_u_hand(hero_idx)

                            if not uh[0][hero_idx]:
                                continue

                            if len(uh[1]) >= hand_street_cutoffs[itr_street_index]:
                                try:
                                    encoded_str = local_enc.encode(json.dumps(uh), True)
                                    buffer_text.append(encoded_str)
                                    buffer_evs.append(hero_ev_data[i])
                                except Exception as e:
                                    continue
                except Exception as e:
                    print(f"Worker {worker_id} Crash: {e}")
                    traceback.print_exc()
                    continue
                finally:
                    if 'hands' in locals(): del hands
                    if 'hero_ev_data' in locals(): del hero_ev_data

            if self.stop_event.is_set(): break

            batch_text = buffer_text[:self.batch_size]
            batch_hero_evs = buffer_evs[:self.batch_size]
            buffer_text = buffer_text[self.batch_size:]
            buffer_evs = buffer_evs[self.batch_size:]

            try:
                self.batch_queue.put({
                    'text': batch_text,
                    'evs': batch_hero_evs,
                    'street_idx': itr_street_index
                }, timeout=1)
            except queue.Full:
                continue

    def rl(self):
        threads = []
        for i in range(self.num_generators):
            t = threading.Thread(target=self.data_generator_worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        losses = []
        itr_cutoff_tokens = [self.river_token, self.turn_token, self.flop_token, self.preflop_token]

        action_map = {
            'fold': self.fold_token.item(),
            'check': self.check_token.item(),
            'call': self.call_token.item(),
            'raise': self.raise_token.item(),
            'allin': self.allin_token.item()
        }
        act_str_to_idx = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3, 'allin': 4}

        print(f"Starting Training with Regret Minimization (Weighted Loss)...")

        try:
            while self.global_updates < 80000:
                try:
                    batch_data = self.batch_queue.get(timeout=300)
                except queue.Empty:
                    print("Queue empty!")
                    break

                batch_text = batch_data['text']
                batch_hero_evs = batch_data['evs']
                itr_street_index = batch_data['street_idx']

                inputs = self.tokenizer(batch_text, padding=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device, non_blocking=True)
                attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
                seq_len = input_ids.shape[1]
                hero_ids = input_ids[:, 1]

                with self.model_lock:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                training_indices = []
                start_index = 26
                q_train = torch.zeros(len(batch_text), dtype=torch.bool).to(self.device)

                for t in range(start_index, seq_len):
                    current_tokens = input_ids[:, t]
                    q_train = q_train | (current_tokens == itr_cutoff_tokens[itr_street_index])
                    is_action_token = ((current_tokens >= 9) & (current_tokens <= 13)) & q_train
                    is_hero_turn = (input_ids[:, t - 1] == hero_ids)
                    target_mask = is_action_token & is_hero_turn

                    if target_mask.any():
                        idxs = torch.nonzero(target_mask).squeeze()
                        if idxs.ndim == 0: idxs = idxs.unsqueeze(0)
                        for idx in idxs:
                            idx_item = idx.item()
                            training_indices.append((idx_item, t))

                if training_indices:
                    all_pred_logits = []
                    all_target_probs = []
                    all_ev_values = []
                    all_valid_masks = []

                    curr_ev_idx = [0] * len(batch_text)

                    for b_idx, t in training_indices:
                        if curr_ev_idx[b_idx] >= len(batch_hero_evs[b_idx]): continue

                        ev_data = batch_hero_evs[b_idx][curr_ev_idx[b_idx]]
                        curr_ev_idx[b_idx] += 1

                        hand_bb = ev_data.get('big_blind', 1.0)
                        temp_divisor = max(5.0 * hand_bb, 1.0)

                        valid_mask = torch.zeros(5, device=self.device)
                        means = np.full(5, -np.inf)
                        ev_values = np.zeros(5)

                        for act_str, samples in ev_data.items():
                            if act_str == 'big_blind': continue
                            if act_str in act_str_to_idx:
                                idx = act_str_to_idx[act_str]
                                if samples:
                                    valid_mask[idx] = 1.0
                                    mean_val = np.mean(samples)
                                    means[idx] = mean_val
                                    ev_values[idx] = mean_val / max(hand_bb, 0.01)

                        valid_means = means[means > -1e9]
                        if len(valid_means) > 0:
                            max_val = np.max(valid_means)
                            exp_vals = np.exp((means - max_val) / temp_divisor)
                            exp_vals[means == -np.inf] = 0
                            probs = exp_vals / np.sum(exp_vals)
                        else:
                            probs = np.zeros(5)

                        relevant_token_ids = [
                            action_map['fold'], action_map['check'], action_map['call'],
                            action_map['raise'], action_map['allin']
                        ]
                        relevant_token_tensor = torch.tensor(relevant_token_ids, device=self.device)
                        current_logits = logits[b_idx, t - 1, relevant_token_tensor]
                        current_logits = current_logits.clone()
                        current_logits[valid_mask == 0] = -100.0

                        all_pred_logits.append(current_logits)
                        all_target_probs.append(torch.tensor(probs, device=self.device, dtype=torch.float32))
                        all_ev_values.append(torch.tensor(ev_values, device=self.device, dtype=torch.float32))
                        all_valid_masks.append(valid_mask)

                    if all_pred_logits:
                        batch_preds = torch.stack(all_pred_logits)
                        batch_targets = torch.stack(all_target_probs)
                        batch_evs = torch.stack(all_ev_values)
                        batch_masks = torch.stack(all_valid_masks)

                        loss = self.regret_loss(batch_preds, batch_targets, batch_evs, batch_masks)

                        losses.append(loss.item())

                        with self.model_lock:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                del logits, outputs, input_ids, attention_mask
                del batch_data, batch_text, batch_hero_evs, inputs

                self.global_updates += 1
                if self.global_updates % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                if self.global_updates % 100 == 0:
                    print(f"Update {self.global_updates} | Loss: {np.mean(losses):.6f} | Q: {self.batch_queue.qsize()}")
                    losses = []
                    with self.model_lock:
                        self.model.eval()
                        torch.save(self.model.state_dict(), f"RL-{self.global_updates}.pt")
                        self.model.train()

        finally:
            self.stop_event.set()
            for t in threads: t.join()

    def batch_generate_hands(self, n_hands, street_cutoff):
        hands = [Hand() for _ in range(n_hands)]
        hero_indices = [randint(0, 5) for _ in range(n_hands)]
        hero_ev_data = [[] for _ in range(n_hands)]
        active_indices = list(range(n_hands))

        while len(active_indices) > 0:
            current_hands = [hands[i] for i in active_indices]
            ev_indices = []
            ev_hand_objs = []
            for i, idx in enumerate(active_indices):
                hand = hands[idx]
                if not hand.done and \
                        hand.state.turn_index == hero_indices[idx] and \
                        hand.state.street_index >= street_cutoff:
                    ev_indices.append(idx)
                    ev_hand_objs.append(hand)

            if len(ev_hand_objs) > 0:
                cf_sizes = self.select_raise_batch(ev_hand_objs)
                bulk_results = self.generate_bulk_evs(ev_hand_objs, cf_sizes)
                for k, global_idx in enumerate(ev_indices):
                    hero_ev_data[global_idx].append(bulk_results[k])

            actions, sizes = self.select_action_batch(current_hands)

            next_active_indices = []
            for i, (action, size) in enumerate(zip(actions, sizes)):
                hand_idx = active_indices[i]
                hand = hands[hand_idx]
                match action:
                    case 'fold':
                        hand.fold()
                    case 'check':
                        hand.check()
                    case 'call':
                        hand.call()
                    case 'raise':
                        hand.bet_or_raise(size)
                    case 'allin':
                        hand.bet_or_raise(size)

                if not hand.done:
                    next_active_indices.append(hand_idx)
            active_indices = next_active_indices

        return hands, hero_ev_data, hero_indices

    @torch.inference_mode()
    def generate_bulk_evs(self, hand_list, size_list):
        all_sims = []
        registry = []
        valid_actions = {'check', 'call', 'raise', 'allin'}

        for i, hand in enumerate(hand_list):
            action_space = hand.get_action_space()
            raise_size = size_list[i]
            player = hand.state.turn_index
            res = {}
            res['big_blind'] = hand.big_blind
            if 'fold' in action_space:
                val = -hand.investment()
                res['fold'] = [val] * self.n_sims
            registry.append({'holder': res, 'is_calc': False})

            for root_action in action_space.keys():
                if root_action == 'fold': continue
                if root_action not in valid_actions: continue

                temp_hand = copy.deepcopy(hand)
                if root_action == 'check':
                    temp_hand.check()
                elif root_action == 'call':
                    temp_hand.call()
                elif root_action == 'raise':
                    if raise_size is None: continue
                    temp_hand.bet_or_raise(raise_size)

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

        while len(active_sim_indices) > 0:
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
                    sim.bet_or_raise(sz)
                elif act == 'raise':
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
                vals = [finished_payoffs[k][plyr] for k in range(start, start + count)]
                current_holder[act] = vals
        if current_holder is not None: final_output.append(current_holder)
        return final_output

    @torch.inference_mode()
    def select_raise_batch(self, hands):
        if not hands: return []

        thread_encoders = self.get_thread_encoders(len(hands))
        results = list(self.executor.map(self.process_hand_cpu, zip(hands, thread_encoders[:len(hands)])))
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, _, _, actor_indices = zip(*results)

        tokenizer = self.get_thread_tokenizer()
        inputs = tokenizer(list(encoded_strs), padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        batch_actor_tokens = self.hero_token_ids[list(actor_indices)].to(self.device)

        raise_col = torch.full((len(hands), 1), self.raise_token_id, device=self.device, dtype=torch.long)

        final_input_ids = torch.cat([input_ids, batch_actor_tokens.unsqueeze(1), raise_col], dim=1)
        extension_mask = torch.ones((len(hands), 2), device=self.device, dtype=torch.long)
        final_attention_mask = torch.cat([attention_mask, extension_mask], dim=1)

        with self.model_lock:
            logits = self.model(final_input_ids, attention_mask=final_attention_mask).logits[:, -1, :]

        start_id = self.min_size_token.item()
        num_sizes = len(self.sizes_floats)
        size_logits = logits[:, start_id: start_id + num_sizes]

        batch_min_tokens = torch.tensor(min_bet_tokens, device=self.device)
        offsets = (batch_min_tokens - start_id).unsqueeze(1)
        size_indices = torch.arange(num_sizes, device=self.device).unsqueeze(0)
        mask = size_indices >= offsets

        size_logits = size_logits.masked_fill(~mask, float('-inf'))
        probs = torch.softmax(size_logits, dim=1)
        relative_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        chosen_percents = self.torch_sizes_float[relative_indices]
        batch_pots = torch.tensor(pot_sizes, device=self.device)
        bets = (batch_pots * (chosen_percents / 100.0)).long()

        final_bets = []
        bets_list = bets.tolist()
        for i, amt in enumerate(bets_list):
            cap = max_bets[i]
            if cap > 0: amt = min(amt, cap)
            final_bets.append(int(amt))
        return final_bets

    @torch.inference_mode()
    def select_action_batch(self, hands):
        if not hands: return [], []

        thread_encoders = self.get_thread_encoders(len(hands))
        results = list(self.executor.map(self.process_hand_cpu, zip(hands, thread_encoders[:len(hands)])))
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, actor_indices = zip(*results)

        tokenizer = self.get_thread_tokenizer()
        inputs = tokenizer(list(encoded_strs), padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        batch_actor_tokens = self.hero_token_ids[list(actor_indices)].to(self.device)

        input_ids = torch.cat([input_ids, batch_actor_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((len(hands), 1), device=self.device)], dim=1)

        with self.model_lock:
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, -1, :]

        action_logits = torch.full_like(logits, float('-inf'))
        mask_check = torch.tensor(can_check, device=self.device)
        mask_raise = torch.tensor(can_raise, device=self.device)

        action_logits[mask_check, self.check_token.item()] = logits[mask_check, self.check_token.item()]
        action_logits[~mask_check, self.fold_token.item()] = logits[~mask_check, self.fold_token.item()]
        action_logits[~mask_check, self.call_token.item()] = logits[~mask_check, self.call_token.item()]
        action_logits[mask_raise, self.raise_token.item()] = logits[mask_raise, self.raise_token.item()]
        action_logits[mask_raise, self.allin_token.item()] = logits[mask_raise, self.allin_token.item()]

        probs = torch.softmax(action_logits, dim=1)
        action_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        action_tokens_cpu = action_tokens.tolist()

        is_raise = False
        for tok in action_tokens_cpu:
            if tok == self.raise_token_id:
                is_raise = True
                break

        final_sizes = [0] * len(hands)
        final_actions = [''] * len(hands)

        for i, tok in enumerate(action_tokens_cpu):
            if tok == self.fold_token.item():
                final_actions[i] = 'fold'
            elif tok == self.check_token.item():
                final_actions[i] = 'check'
            elif tok == self.call_token.item():
                final_actions[i] = 'call'
            elif tok == self.allin_token.item():
                final_actions[i] = 'allin'
                final_sizes[i] = max_bets[i]
            elif tok == self.raise_token.item():
                final_actions[i] = 'raise'

        if is_raise:
            raise_idxs = [i for i, x in enumerate(final_actions) if x == 'raise']
            if raise_idxs:
                r_encoded = [encoded_strs[i] for i in raise_idxs]

                tokenizer = self.get_thread_tokenizer()
                r_inputs = tokenizer(r_encoded, padding=True, return_tensors="pt")
                r_ids = r_inputs.input_ids.to(self.device, non_blocking=True)
                r_mask = r_inputs.attention_mask.to(self.device, non_blocking=True)

                r_actors = self.hero_token_ids[list(np.array(actor_indices)[raise_idxs])].to(self.device)

                r_raise_tok = torch.full((len(raise_idxs), 1), self.raise_token_id, device=self.device)
                r_final_ids = torch.cat([r_ids, r_actors.unsqueeze(1), r_raise_tok], dim=1)
                r_ext_mask = torch.ones((len(raise_idxs), 2), device=self.device)
                r_final_mask = torch.cat([r_mask, r_ext_mask], dim=1)

                with self.model_lock:
                    r_logits = self.model(r_final_ids, attention_mask=r_final_mask).logits[:, -1, :]

                start_id = self.min_size_token.item()
                num_sizes = len(self.sizes_floats)
                size_logits = r_logits[:, start_id: start_id + num_sizes]
                r_min_tokens = torch.tensor([min_bet_tokens[i] for i in raise_idxs], device=self.device)
                offsets = (r_min_tokens - start_id).unsqueeze(1)
                size_indices = torch.arange(num_sizes, device=self.device).unsqueeze(0)
                mask = size_indices >= offsets
                size_logits = size_logits.masked_fill(~mask, float('-inf'))
                size_probs = torch.softmax(size_logits, dim=1)
                r_indices = torch.multinomial(size_probs, num_samples=1).squeeze(1)
                r_chosen_pct = self.torch_sizes_float[r_indices]
                r_pots = torch.tensor([pot_sizes[i] for i in raise_idxs], device=self.device)
                r_bets = (r_pots * (r_chosen_pct / 100.0)).long()
                r_bets_cpu = r_bets.tolist()
                for k, hand_idx in enumerate(raise_idxs):
                    final_sizes[hand_idx] = int(min(r_bets_cpu[k], max_bets[hand_idx]))

        return final_actions, final_sizes

    def process_hand_cpu(self, args):
        hand, encoder = args
        action_space = hand.get_action_space()
        pot_size = hand.pot_size()
        if 'min_bet' in action_space:
            min_bet = action_space['min_bet']
            target_pct = (min_bet / pot_size) * 100
            idx = np.searchsorted(self.sizes_floats, target_pct, side='left')
            if idx < len(self.sizes_floats):
                min_bet_token = self.min_size_token + idx
            else:
                min_bet_token = self.min_size_token + (len(self.sizes_floats) - 1)
            max_bet = action_space['max_bet']
        else:
            min_bet_token = 0
            max_bet = 0

        turn_idx = hand.state.turn_index
        u_hand = hand.get_u_hand(turn_idx)
        encoded_str = encoder.encode(json.dumps(u_hand))
        can_check = 'check' in action_space
        can_raise = 'min_bet' in action_space
        return encoded_str, min_bet_token, max_bet, pot_size, can_check, can_raise, turn_idx


if __name__ == '__main__':
    sim = Simulator()
    sim.rl()
