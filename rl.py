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
import torch.multiprocessing as mp
import torch.nn.functional as F
import schedulefree
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from sim_hand import Hand
from sim_encoder import Encoder
import threading
import queue
import concurrent.futures


class ThreadSafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.value

    def increment(self):
        with self.lock:
            self.value += 1

class Simulator:
    def __init__(self):
        self.device = 'cuda'
        torch.set_float32_matmul_precision('high')
        self.thread_local = threading.local()

        config = AutoConfig.from_pretrained('./config.json')

        self.model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.model.share_memory()

        self.ref_model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.ref_model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.ref_model.share_memory()


        base_tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        base_tokenizer.padding_side = "left"
        base_tokenizer.pad_token = base_tokenizer.unk_token

        self.fold_token_id = base_tokenizer.encode("<xxx>")[0]
        self.check_token_id = base_tokenizer.encode("<xxx>")[0]
        self.call_token_id = base_tokenizer.encode("<xxx>")[0]
        self.raise_token_id = base_tokenizer.encode("<xxx>")[0]
        self.allin_token_id = base_tokenizer.encode("<xxx>")[0]

        self.min_size_token_id = base_tokenizer.encode("<xxx>")[0]
        self.min_size_token = torch.tensor([self.min_size_token_id]).to(self.device)

        self.hero_token_ids = torch.tensor([base_tokenizer.encode(f"<xxx>")[0] for i in range(6)],
                                           device=self.device)
        self.action_tokens = {
            'fold': self.fold_token_id,
            'check': self.check_token_id,
            'call': self.call_token_id,
            'raise': self.raise_token_id,
            'allin': self.allin_token_id
        }

        self.sizes = np.array(list(range(1, 5))..., dtype=np.float32)
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()

        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=1e-6,
            warmup_steps=0,
            betas=(0.9, 0.999),
            weight_decay=0.0
        )
        self.optimizer.train()

        self.n_sims = 8
        self.batch_size = 8
        self.global_updates = 0



    @property
    def tokenizer(self):
        if not hasattr(self.thread_local, 'tokenizer'):
            tok = AutoTokenizer.from_pretrained('./opt-it-2')
            tok.padding_side = "left"
            tok.pad_token = tok.unk_token
            self.thread_local.tokenizer = tok
        return self.thread_local.tokenizer

    @property
    def encoder(self):
        if not hasattr(self.thread_local, 'encoder'):
            self.thread_local.encoder = Encoder()
        return self.thread_local.encoder

    @property
    def encoders(self):
        if not hasattr(self.thread_local, 'encoders'):
            encoders = []
            for i in range(self.batch_size):
                encoders.append(Encoder())
            self.thread_local.encoders = encoders
        return self.thread_local.encoders

    def generate_experience(self, current_train_count):
        hands, hero_ev_data, hero_indices, full_seqs = self.batch_generate_hands(self.batch_size)
        experiences = []

        for i, hand in enumerate(hands):
            if not hero_ev_data[i]: continue

            big_blind = float(hand.big_blind) if hand.big_blind > 0 else 1.0

            hand_decision_counts = []
            hand_legal_actions = []
            hand_ev_scores = []

            for node_data in hero_ev_data[i]:
                ev_dict = node_data['evs']

                means = {
                    act_str: float(np.mean(samples)) / big_blind
                    for act_str, samples in ev_dict.items()
                    if act_str in self.action_tokens and samples
                }

                if not means: continue

                acts = list(means.keys())
                scores = np.array([means[a] for a in acts])

                hand_decision_counts.append(node_data['decision_token_count'])
                hand_legal_actions.append(acts)
                hand_ev_scores.append(scores.tolist())

            if hand_decision_counts:
                experience = {
                    'full_text': full_seqs[i],
                    'decision_token_counts': hand_decision_counts,
                    'legal_actions': hand_legal_actions,
                    'ev_scores': hand_ev_scores,
                    'train_count': current_train_count
                }
                experiences.append(experience)

        return experiences

    def worker_loop(self, worker_id, q, train_count_obj):
        print(f"Generator Thread {worker_id} started.")
        self.model.eval()

        while True:
            try:
                current_tc = train_count_obj.get()

                with torch.no_grad():
                    experiences = self.generate_experience(current_tc)

                for exp in experiences:
                    q.put(exp)
            except Exception as e:
                print(f"Generator {worker_id} Exception: {e}")
                traceback.print_exc()

    def run_trainer(self, q, train_count_obj):
        print("Starting Central Trainer Loop...")

        log_file = open('training_logs.csv', mode='w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Update', 'Strategy_Loss', 'KL_Loss', 'Penalty_Loss'])

        batch_memory = []

        while self.global_updates < 40000:

            while len(batch_memory) < self.batch_size:
                try:
                    exp = q.get(timeout=1)
                    if exp['train_count'] == train_count_obj.get():
                        batch_memory.append(exp)
                except queue.Empty:
                    continue

            batch_samples = batch_memory[:self.batch_size]
            batch_memory = batch_memory[self.batch_size:]

            full_texts = [s['full_text'] for s in batch_samples]

            inputs = self.tokenizer(full_texts, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device, non_blocking=True)
            attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

            seq_len = input_ids.size(1)
            unpadded_lengths_cpu = attention_mask.sum(dim=1).long().tolist()

            batch_b_indices = []
            batch_seq_indices = []
            flat_legal_actions = []
            flat_ev_scores = []

            for b in range(self.batch_size):
                counts = batch_samples[b]['decision_token_counts']
                for i, count in enumerate(counts):
                    pad_offset = seq_len - unpadded_lengths_cpu[b]
                    token_idx = pad_offset + count - 1

                    batch_b_indices.append(b)
                    batch_seq_indices.append(token_idx)
                    flat_legal_actions.append(batch_samples[b]['legal_actions'][i])
                    flat_ev_scores.append(batch_samples[b]['ev_scores'][i])

            b_idx_tensor = torch.tensor(batch_b_indices, device=self.device, dtype=torch.long)
            seq_idx_tensor = torch.tensor(batch_seq_indices, device=self.device, dtype=torch.long)
            total_nodes = len(batch_b_indices)

            self.model.train()
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                active_logits = self.model(input_ids, attention_mask=attention_mask).logits.float()

                with torch.no_grad():
                    ref_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits.float()

            s_preds = active_logits[b_idx_tensor, seq_idx_tensor, :]
            s_targets = torch.zeros_like(s_preds)
            batch_illegal_mask = torch.ones_like(s_preds, dtype=torch.bool)
            dynamic_weights = []

            with torch.no_grad():
                max_acts = 5
                padded_ids = np.zeros((total_nodes, max_acts), dtype=np.int64)
                padded_scores = np.zeros((total_nodes, max_acts), dtype=np.float32)
                valid_mask = np.zeros((total_nodes, max_acts), dtype=bool)

                valid_b_indices = []
                valid_act_indices = []

                for n in range(total_nodes):
                    acts = flat_legal_actions[n]
                    scores = flat_ev_scores[n]
                    local_ids = [self.action_tokens[act] for act in acts]
                    num_acts = len(acts)

                    padded_ids[n, :num_acts] = local_ids
                    padded_ids[n, num_acts:] = local_ids[0]
                    padded_scores[n, :num_acts] = scores
                    valid_mask[n, :num_acts] = True

                    valid_b_indices.extend([n] * num_acts)
                    valid_act_indices.extend(local_ids)

                    spread = max(scores) - min(scores)
                    dynamic_weights.append(float(spread) + 1e-3)

                act_ids_tensor = torch.tensor(padded_ids, device=self.device, dtype=torch.long)
                scores_tensor = torch.tensor(padded_scores, device=self.device, dtype=torch.float32)
                valid_mask_tensor = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)

                batch_illegal_mask[valid_b_indices, valid_act_indices] = False

                current_probs_full = F.softmax(s_preds, dim=-1)
                policy_probs = torch.gather(current_probs_full, 1, act_ids_tensor)
                policy_probs = policy_probs * valid_mask_tensor.float()

                prob_sums = policy_probs.sum(dim=-1, keepdim=True)
                valid_counts = valid_mask_tensor.sum(dim=-1, keepdim=True).float()
                uniform_probs = valid_mask_tensor.float() / valid_counts
                policy_probs = torch.where(prob_sums > 0, policy_probs / prob_sums, uniform_probs)

                policy_ev = torch.sum(policy_probs * scores_tensor, dim=-1, keepdim=True)
                regrets = scores_tensor - policy_ev
                pos_regrets = torch.clamp(regrets, min=0.0) * valid_mask_tensor.float()

                sum_pos_regrets = torch.sum(pos_regrets, dim=-1, keepdim=True)
                target_probs = torch.where(sum_pos_regrets > 0, pos_regrets / sum_pos_regrets, policy_probs)

                masked_targets = target_probs * valid_mask_tensor.float()
                s_targets.scatter_add_(1, act_ids_tensor, masked_targets)

            s_preds_masked = torch.where(
                batch_illegal_mask,
                torch.tensor(-1000.0, device=self.device, dtype=s_preds.dtype),
                s_preds
            )

            illegal_logits = s_preds[batch_illegal_mask]
            violation = F.relu(illegal_logits - (-5.0))
            if violation.numel() > 0:
                illegal_penalty_loss = violation.mean()
            else:
                illegal_penalty_loss = torch.tensor(0.0, device=self.device)
                
            s_loss_unreduced = F.cross_entropy(s_preds_masked, s_targets, reduction='none')

            batch_weights = torch.tensor(dynamic_weights, device=self.device)
            batch_weights = batch_weights / batch_weights.mean()
            s_loss = torch.mean(s_loss_unreduced * batch_weights)

            kl_unreduced = F.kl_div(
                F.log_softmax(active_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='none'
            ).sum(dim=-1)

            masked_kl = kl_unreduced * attention_mask
            seq_kl = masked_kl.sum() / attention_mask.sum()

            decision_kl = kl_unreduced[b_idx_tensor, seq_idx_tensor].mean()

            kl_loss = seq_kl + decision_kl

            total_loss = s_loss + kl_loss + (0.1 * illegal_penalty_loss)

            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_count_obj.increment()
            self.global_updates += 1

            if self.global_updates % 10 == 0:
                csv_writer.writerow([self.global_updates, s_loss.item(), kl_loss.item(), illegal_penalty_loss.item()])
                log_file.flush()

            if self.global_updates % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Upd {self.global_updates} | Nodes: {total_nodes} | Strat: {s_loss.item():.4f} | KL: {kl_loss.item():.4f} | Pen: {illegal_penalty_loss.item():.4f}")
                torch.save(self.model.state_dict(), f"RL-{self.global_updates}.pt")

    def batch_generate_hands(self, n_hands):
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
                if not hand.done and hand.state.turn_index == hero_indices[idx]:
                    ev_indices.append(idx)
                    ev_hand_objs.append(hand)

            if ev_hand_objs:
                cf_sizes = self.select_raise_batch(ev_hand_objs)
                bulk_results = self.generate_bulk_evs(ev_hand_objs, cf_sizes)
                for k, global_idx in enumerate(ev_indices):
                    if hero_indices[global_idx] != ev_hand_objs[k].state.turn_index:
                        print("turn mismatch")
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

        valid_indices = [i for i, evs in enumerate(hero_ev_data) if evs]
        full_seqs = [""] * len(hands)

        if valid_indices:
            current_encoders = self.encoders

            def _process_entire_trajectory(encoder, hand, ev_nodes, hero_idx):
                final_seq = encoder.encode(json.dumps(hand.get_u_hand(hero_idx)), True, True)
                valid_nodes = []
                for node in ev_nodes:
                    state_json = node['state_json']

                    hist_encoded = encoder.encode(state_json)
                    decision_token_count = f"{hist_encoded}<herop{hero_idx}>".count('<')

                    valid_nodes.append({
                        'decision_token_count': decision_token_count,
                        'evs': node['evs']
                    })

                return final_seq, valid_nodes

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_indices)) as executor:
                futures = [
                    executor.submit(
                        _process_entire_trajectory,
                        current_encoders[idx],
                        hands[idx],
                        hero_ev_data[idx],
                        hero_indices[idx]
                    )
                    for idx in valid_indices
                ]

                for idx, future in zip(valid_indices, futures):
                    final_seq, valid_nodes = future.result()
                    full_seqs[idx] = final_seq
                    hero_ev_data[idx] = valid_nodes

        return hands, hero_ev_data, hero_indices, full_seqs

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

            can_raise = 'min_bet' in action_space
            if can_raise and action_space['min_bet'] >= max_bet:
                can_raise = False

            can_call = 'call' in action_space
            call_is_allin = can_call and not can_raise

            if not can_raise:
                valid_actions.discard('min_bet')

            if call_is_allin:
                valid_actions.discard('max_bet')

            if max_bet > 4 * raise_size:
                valid_actions.discard('max_bet')
            elif max_bet > 0 and raise_size >= (0.5 * max_bet):
                valid_actions.discard('min_bet')

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

            can_raise = 'min_bet' in action_space
            if can_raise and action_space['min_bet'] >= max_bet:
                can_raise = False

            can_call = 'call' in action_space
            call_is_allin = can_call and not can_raise

            if 'min_bet' in final_output[i]:
                final_output[i]['raise'] = final_output[i].pop('min_bet')
            if 'max_bet' in final_output[i]:
                final_output[i]['allin'] = final_output[i].pop('max_bet')

            if call_is_allin and 'call' in final_output[i]:
                final_output[i]['allin'] = final_output[i].pop('call')

        return final_output


    @torch.inference_mode()
    def select_raise_batch(self, hands):
        if not hands: return []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, _, _, _, _, actor_indices = zip(*results)

        queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}><raise>" for i in range(len(hands))]

        inputs = self.tokenizer(queries, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        logits = self.model(input_ids, attention_mask=attention_mask).logits

        start_id = self.min_size_token.item()
        num_sizes = len(self.sizes_floats)

        size_logits = logits[:, -1, start_id: start_id + num_sizes]
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

        return final_bets

    @torch.inference_mode()
    def select_action_batch(self, hands):
        if not hands: return [], []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices = zip(
            *results)

        queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}>" for i in range(len(hands))]

        inputs = self.tokenizer(queries, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        logits = self.model(input_ids, attention_mask=attention_mask).logits

        action_keys = ['fold', 'check', 'call', 'raise', 'allin']
        action_ids = [self.fold_token_id, self.check_token_id, self.call_token_id, self.raise_token_id,
                      self.allin_token_id]
        action_tensor = torch.tensor(action_ids, device=self.device)

        action_logits = logits[:, -1, action_tensor]
        final_actions = [''] * len(hands)
        final_sizes = [0] * len(hands)
        raise_indexes = []

        action_probs = F.softmax(action_logits, dim=-1)

        legal_mask_tensor = torch.stack([
            torch.tensor([
                not can_check[i],
                can_check[i],
                can_call[i] and not call_is_allin[i],
                can_raise[i],
                max_bets[i] > 0 or call_is_allin[i]
            ], device=self.device) for i in range(len(hands))
        ])

        action_probs = action_probs * legal_mask_tensor.float()
        sum_probs = action_probs.sum(dim=-1, keepdim=True)
        action_probs = torch.where(sum_probs > 0, action_probs / sum_probs,
                                   legal_mask_tensor.float() / legal_mask_tensor.sum(dim=-1, keepdim=True))

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

            r_queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}><raise>" for i in raise_indexes]

            r_inputs = self.tokenizer(r_queries, padding=True, return_tensors="pt")

            r_final_ids = r_inputs.input_ids.to(self.device, non_blocking=True)
            r_final_mask = r_inputs.attention_mask.to(self.device, non_blocking=True)

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
            min_bet_token, max_bet, pot_size, can_check, can_raise, can_call, call_is_allin, turn_idx
        )

if __name__ == '__main__':
    sim = Simulator()

    experience_queue = queue.Queue(maxsize=sim.batch_size * 2)
    global_train_count = ThreadSafeCounter()

    num_generators = 4
    threads = []

    for i in range(num_generators):
        t = threading.Thread(
            target=sim.worker_loop,
            args=(i, experience_queue, global_train_count),
            daemon=True
        )
        t.start()
        threads.append(t)

    try:
        sim.run_trainer(experience_queue, global_train_count)
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
