import sys
import os

sys.modules["markupsafe._speedups"] = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import json
import queue
import threading
import concurrent.futures
from random import randint
import traceback
import torch
import schedulefree
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import Counter
import gc

from sim_hand import Hand
from sim_encoder import Encoder

class Wasserstein1DLoss(torch.nn.Module):
    def __init__(self, device, min_return=-50, max_return=150, n_atoms=51):
        super().__init__()
        self.device = device
        # The physical cost of moving mass 1 bin (e.g. 4 BBs)
        self.delta_z = (max_return - min_return) / (n_atoms - 1)

    def forward(self, pred_logits, target_probs):
        # Softmax to get PDF
        pred_probs = torch.softmax(pred_logits, dim=-1)

        # Cumulative Sum to get CDF (The "Volume" of mass)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)

        # Exact 1D Wasserstein: Area between CDF curves
        cdf_diff = torch.abs(pred_cdf - target_cdf)

        # Sum (Integrate) and scale by bucket distance
        return torch.sum(cdf_diff, dim=-1).mean() * self.delta_z


class DistributionalOTUtils:
    def __init__(self, device, min_return_bb=-50, max_return_bb=150, n_atoms=51):
        self.device = device
        self.n_atoms = n_atoms
        self.min_return = min_return_bb
        self.max_return = max_return_bb
        self.support = torch.linspace(min_return_bb, max_return_bb, n_atoms).to(device)
        self.delta_z = (max_return_bb - min_return_bb) / (n_atoms - 1)

    def to_categorical(self, rewards_list, big_blinds):
        batch_size = len(rewards_list)
        distributions = torch.zeros(batch_size, self.n_atoms, device=self.device)

        for i, r in enumerate(rewards_list):
            if r is None:
                zero_idx = (torch.abs(self.support)).argmin()
                distributions[i, zero_idx] = 1.0
                continue

            bb = big_blinds[i].item() if big_blinds is not None else 1.0
            if bb < 1e-3: bb = 1.0

            # Use mean of simulation samples as the target value
            val = np.mean(r)
            r_tensor = torch.tensor(val, device=self.device, dtype=torch.float32) / bb
            r_tensor = torch.clamp(r_tensor, self.min_return, self.max_return)

            # Projection logic (Dual Splitting)
            b = (r_tensor - self.min_return) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            l = torch.clamp(l, 0, self.n_atoms - 1)
            u = torch.clamp(u, 0, self.n_atoms - 1)

            exact_match = (l == u)
            distributions[i, l] += torch.where(exact_match, 1.0, u.float() - b)
            distributions[i, u] += torch.where(exact_match, 0.0, b - l.float())

        return distributions

class Simulator:
    def __init__(self):
        self.device = 'cuda'
        config = AutoConfig.from_pretrained('./config.json')
        self.model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))

        # Predicts the outcome histogram (51 atoms) from the hidden state
        self.n_atoms = 31
        self.dist_head = torch.nn.Linear(config.hidden_size, self.n_atoms).to(self.device)
        self.dist_head.weight.data.normal_(mean=0.0, std=0.02)

        self.ref_model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.ref_model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.model_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # Hero Tokens
        hero_ids = []
        for i in range(6):
            enc = self.tokenizer.encode(f"xxx{i}x")
            hero_ids.append(enc[0])
        self.hero_token_ids = torch.tensor(hero_ids, dtype=torch.long, device=self.device)

        # Tokens
        self.result_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.fold_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.check_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.call_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.raise_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.raise_token_id = self.raise_token.item()
        self.allin_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.min_size_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.preflop_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.flop_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.turn_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)
        self.river_token = torch.tensor(self.tokenizer.encode("<xxx>")).to(self.device)

        self.sizes = np.array(list(range(1, 5)) + list(range(5, 101, 5)) + list(range(125, 501, 25)), dtype=np.float32)
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()

        self.action_token_ids = torch.tensor([
            self.fold_token.item(), self.check_token.item(),
            self.call_token.item(), self.raise_token.item(),
            self.allin_token.item()
        ], device=self.device)

        # Loss Functions
        self.ot_utils = DistributionalOTUtils(self.device, min_return_bb=-200, max_return_bb=500, n_atoms=self.n_atoms)
        self.wasserstein_loss = Wasserstein1DLoss(self.device, min_return=-200, max_return=500, n_atoms=self.n_atoms)
        self.policy_loss = torch.nn.CrossEntropyLoss()

        # Optimizer: Tracks both LLM and Distribution Head
        params = list(self.model.parameters()) + list(self.dist_head.parameters())
        self.optimizer = schedulefree.AdamWScheduleFree(params, lr=1e-5)
        self.optimizer.train()

        self.n_sims = 16
        self.batch_size = 256
        self.num_generators = 5
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
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
                        self.batch_size * 2, itr_street_cutoffs[itr_street_index]
                    )
                    local_enc = self.get_single_thread_encoder()
                    for i, hand in enumerate(hands):
                        if len(hero_ev_data[i]) > 0:
                            hero_idx = hero_indices[i]
                            uh = hand.get_u_hand(hero_idx)
                            if not uh[0][hero_idx]: continue
                            if len(uh[1]) >= hand_street_cutoffs[itr_street_index]:
                                try:
                                    encoded_str = local_enc.encode(json.dumps(uh), True)
                                    buffer_text.append(encoded_str)
                                    buffer_evs.append(hero_ev_data[i])
                                except:
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
                self.batch_queue.put({'text': batch_text, 'evs': batch_hero_evs, 'street_idx': itr_street_index}, timeout=1)
            except queue.Full:
                continue

    def rl(self):
        # Initialize and Track Threads
        threads = []
        for i in range(self.num_generators):
            t = threading.Thread(target=self.data_generator_worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        losses = []
        ot_losses = []
        policy_losses = []

        # Define Params for Clipping
        params = list(self.model.parameters()) + list(self.dist_head.parameters())

        itr_cutoff_tokens = [self.river_token, self.turn_token, self.flop_token, self.preflop_token]
        act_str_to_idx = {'fold': 0, 'check': 1, 'call': 2, 'raise': 3, 'allin': 4}

        print(f"Starting Joint Training: Policy (CE) + Outcome (Wasserstein)...")

        try:
            while self.global_updates < 80000:
                try:
                    batch_data = self.batch_queue.get(timeout=300)
                except queue.Empty:
                    print("Queue Empty - breaking")
                    break

                batch_text = batch_data['text']
                batch_hero_evs = batch_data['evs']
                itr_street_index = batch_data['street_idx']

                inputs = self.tokenizer(batch_text, padding=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device, non_blocking=True)
                attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
                seq_len = input_ids.shape[1]

                with self.model_lock:
                    # Output hidden states for the Distrib Head
                    outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    logits = outputs.logits
                    last_hidden = outputs.hidden_states[-1]

                # Forward pass on frozen reference model for sizing KL penalty
                with torch.no_grad():
                    ref_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits

                # Offset by 1 because logits at t-1 predict token at t
                current_tokens = input_ids[:, 1:]
                prev_tokens = input_ids[:, :-1]

                # Mask out any transitions involving pad tokens (either as input or target)
                valid_mask = (current_tokens != self.tokenizer.pad_token_id) & \
                             (prev_tokens != self.tokenizer.pad_token_id)

                # Dynamic hero token indexing
                non_pad_mask = input_ids != self.tokenizer.pad_token_id
                first_non_pad_indices = non_pad_mask.int().argmax(dim=1)
                hero_token_indices = torch.clamp(first_non_pad_indices + 1, max=seq_len - 1)
                hero_ids_batch = input_ids[torch.arange(input_ids.shape[0]), hero_token_indices]

                # q_train cumulative mask
                cutoff_token = itr_cutoff_tokens[itr_street_index]
                is_cutoff = (current_tokens == cutoff_token)
                q_train = is_cutoff.cumsum(dim=1) > 0

                is_hero_turn = (prev_tokens == hero_ids_batch.unsqueeze(1))
                is_action_token = torch.isin(current_tokens, self.action_token_ids)

                # Partition the valid tokens
                hero_action_mask = is_action_token & is_hero_turn & q_train & valid_mask
                other_mask = valid_mask & (~hero_action_mask)
                hero_b_idx, hero_t_idx = torch.nonzero(hero_action_mask, as_tuple=True)

                all_pred_logits = []
                all_target_probs = []
                all_pred_distribs = []
                all_target_distribs = []
                all_bbs = []

                curr_ev_idx = [0] * len(batch_text)

                for i in range(len(hero_b_idx)):
                    b = hero_b_idx[i].item()
                    t_minus_1 = hero_t_idx[i].item()

                    if curr_ev_idx[b] >= len(batch_hero_evs[b]): continue
                    ev_data = batch_hero_evs[b][curr_ev_idx[b]]
                    curr_ev_idx[b] += 1

                    hand_bb = ev_data.get('big_blind', 1.0)
                    valid_action_mask = torch.zeros(5, device=self.device)
                    means = np.full(5, -np.inf)
                    best_action_samples = None

                    for act_str, samples in ev_data.items():
                        if act_str == 'big_blind': continue
                        if act_str in act_str_to_idx:
                            idx = act_str_to_idx[act_str]
                            if samples:
                                valid_action_mask[idx] = 1.0
                                means[idx] = np.mean(samples)
                                if best_action_samples is None or means[idx] > np.max(means[means > -1e9]):
                                    best_action_samples = samples

                    valid_means = means[means > -1e9]
                    if len(valid_means) > 0:
                        max_val = np.max(valid_means)
                        std_dev = np.std(valid_means) if len(valid_means) > 1 else 0.0
                        temp_divisor = max(std_dev, hand_bb * 1.0, 1e-3)
                        exp_vals = np.exp((means - max_val) / temp_divisor)
                        exp_vals[means == -np.inf] = 0
                        probs = exp_vals / np.sum(exp_vals)
                    else:
                        probs = np.zeros(5)

                    if best_action_samples:
                        all_target_distribs.append(best_action_samples)
                        all_bbs.append(hand_bb)
                        current_dist_pred = self.dist_head(last_hidden[b, t_minus_1, :])
                        all_pred_distribs.append(current_dist_pred)



                    # Slice specific valid logits
                    current_logits = logits[b, t_minus_1, self.action_token_ids].clone()
                    current_logits[valid_action_mask == 0] = -100.0
                    all_pred_logits.append(current_logits)
                    all_target_probs.append(torch.tensor(probs, device=self.device, dtype=torch.float32))

                p_loss = torch.tensor(0.0, device=self.device)
                ot_loss = torch.tensor(0.0, device=self.device)

                if all_pred_logits:
                    batch_preds = torch.stack(all_pred_logits)
                    batch_targets = torch.stack(all_target_probs)
                    p_loss = self.policy_loss(batch_preds, batch_targets)

                    if all_pred_distribs:
                        pred_dists = torch.stack(all_pred_distribs)
                        bb_tensor = torch.tensor(all_bbs, device=self.device, dtype=torch.float32)
                        target_hists = self.ot_utils.to_categorical(all_target_distribs, bb_tensor)
                        ot_loss = self.wasserstein_loss(pred_dists, target_hists)

                other_b_idx, other_t_idx = torch.nonzero(other_mask, as_tuple=True)
                kl_loss = torch.tensor(0.0, device=self.device)

                if len(other_b_idx) > 0:
                    # Slice the full vocabulary distributions
                    sliced_logits_train = logits[other_b_idx, other_t_idx, :]
                    sliced_logits_ref = ref_logits[other_b_idx, other_t_idx, :]

                    log_probs_train = F.log_softmax(sliced_logits_train, dim=-1)
                    probs_ref = F.softmax(sliced_logits_ref, dim=-1)

                    # Batchmean handles sequence length variations cleanly
                    kl_loss = F.kl_div(log_probs_train, probs_ref, reduction='batchmean')
                    
                total_loss = p_loss + (ot_loss / (self.n_atoms * 2.5)) + (0.1 * kl_loss)

                if total_loss.requires_grad:
                    losses.append(total_loss.item())
                    policy_losses.append(p_loss.item())
                    ot_losses.append(ot_loss.item())

                    with self.model_lock:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                del logits, ref_logits, outputs, input_ids, attention_mask, last_hidden
                del batch_data, batch_text, batch_hero_evs, inputs

                self.global_updates += 1
                if self.global_updates % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                if self.global_updates % 1000 == 0:
                    p_mean = np.mean(policy_losses) if policy_losses else 0
                    ot_mean = np.mean(ot_losses) if ot_losses else 0
                    print(f"Upd {self.global_updates} | Loss: {np.mean(losses):.4f} (Pol: {p_mean:.4f}, OT: {ot_mean:.4f}) | Q: {self.batch_queue.qsize()}")
                    losses, policy_losses, ot_losses = [], [], []
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
                if not hand.done and hand.state.turn_index == hero_indices[idx] and hand.state.street_index >= street_cutoff:
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
                if action == 'fold':
                    hand.fold()
                elif action == 'check':
                    hand.check()
                elif action == 'call':
                    hand.call()
                elif action == 'raise':
                    hand.bet_or_raise(size)
                elif action == 'allin':
                    hand.bet_or_raise(size)
                if not hand.done: next_active_indices.append(hand_idx)
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
        final_sizes = [0] * len(hands)
        final_actions = [''] * len(hands)
        is_raise = False
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
                is_raise = True
            elif tok == self.raise_token.item():
                final_actions[i] = 'raise'
                is_raise = True
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
