import sys
import os
import copy
import json
import queue
import threading
import concurrent.futures
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

        self.model_lock = threading.Lock()
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

        self.min_size_token_id = self.tokenizer.encode("<xxx>")[0]
        self.min_size_token = torch.tensor([self.min_size_token_id]).to(self.device)

        # Core Tokens
        self.hero_token_ids = torch.tensor([self.tokenizer.encode(f"<xxx{i}>")[0] for i in range(6)], device=self.device)
        self.action_tokens = {
            'fold': self.fold_token_id,
            'check': self.check_token_id,
            'call': self.call_token_id,
            'raise': self.raise_token_id,
            'allin': self.allin_token_id
        }

        # Token-Space Regret Supports
        self.sizes = np.array(list(range(1, 5)) #...
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()

        # Optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(), lr=1e-6, warmup_steps=0, betas=(0.9, 0.999), weight_decay=0.01
        )
        self.optimizer.train()

        # Engine Params
        self.n_sims = 16
        self.batch_size = 32
        self.num_generators = 4
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.thread_local = threading.local()
        self.batch_queue = queue.Queue(maxsize=16)
        self.stop_event = threading.Event()
        self.global_updates = 0

        self.log_file = open('training_logs.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['Update', 'Total_Loss', 'Strategy_Loss', 'Regret_Loss', 'Queue_Size'])

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

        buffer_strategy_text, buffer_strategy_targets = [], []
        buffer_regret_text, buffer_regret_targets = [], []

        while not self.stop_event.is_set():
            updates = self.global_updates
            itr_street_index = min((updates // 20000), len(itr_street_cutoffs) - 1)

            while (len(buffer_strategy_text) < self.batch_size or len(buffer_regret_text) < self.batch_size) and not self.stop_event.is_set():
                try:
                    hands, hero_ev_data, hero_indices = self.batch_generate_hands(
                        self.batch_size, itr_street_cutoffs[itr_street_index]
                    )
                    local_enc = self.get_single_thread_encoder()

                    for i, hand in enumerate(hands):
                        if not hero_ev_data[i]: continue

                        hero_idx = hero_indices[i]
                        uh = hand.get_u_hand(hero_idx)
                        big_blind = float(hand.big_blind) if hand.big_blind > 0 else 1.0

                        if not uh[0][hero_idx] or len(uh[1]) < hand_street_cutoffs[itr_street_index]:
                            continue

                        encoded_str = local_enc.encode(json.dumps(uh))
                        base_encoded = f"{encoded_str}<herop{hero_idx}>"

                        for ev_dict in hero_ev_data[i]:
                            means = {
                                act_str: float(np.mean(samples)) / big_blind
                                for act_str, samples in ev_dict.items()
                                if act_str in self.action_tokens and samples
                            }

                            if not means: continue

                            acts = list(means.keys())
                            scores = np.array([means[a] for a in acts])
                            max_score = np.max(scores)

                            exp_vals = np.exp((scores - max_score) / big_blind)
                            probs = exp_vals / np.sum(exp_vals)

                            strategy_target = {acts[j]: float(probs[j]) for j in range(len(acts))}
                            total_prob = sum(strategy_target.values())
                            strategy_target = {k: v / total_prob for k, v in strategy_target.items()}

                            #v_I = sum(strategy_target[a] * means[a] for a in acts)

                            buffer_strategy_text.append(base_encoded)
                            buffer_strategy_targets.append(strategy_target)

                            for act in acts:
                                buffer_regret_text.append(f"{base_encoded}<{act}><unk>")
                                buffer_regret_targets.append(max_score - means[act])

                except Exception as e:
                    print(f"Worker {worker_id} Crash: {e}")
                    traceback.print_exc()
                    continue

            if self.stop_event.is_set(): break

            # Safely extract and clear exact batch sizes to prevent memory fragmentation
            batch_data = {
                'strategy_text': [buffer_strategy_text.pop(0) for _ in range(self.batch_size)],
                'strategy_targets': [buffer_strategy_targets.pop(0) for _ in range(self.batch_size)],
                'regret_text': [buffer_regret_text.pop(0) for _ in range(self.batch_size)],
                'regret_targets': [buffer_regret_targets.pop(0) for _ in range(self.batch_size)]
            }

            while not self.stop_event.is_set():
                try:
                    self.batch_queue.put(batch_data, timeout=1)
                    break
                except queue.Full:
                    continue

    def rl(self):
        threads = [threading.Thread(target=self.data_generator_worker, args=(i,), daemon=True) for i in range(self.num_generators)]
        for t in threads: t.start()

        print("Starting Joint Training: Deep CFR Strategy + Token-Space Regret + KL Div...")
        beta = 1.0

        try:
            while self.global_updates < 80000:
                try:
                    batch_data = self.batch_queue.get(timeout=300)
                except queue.Empty:
                    print("Queue Empty - breaking")
                    break

                # Strategy & KL
                strat_inputs = self.tokenizer(batch_data['strategy_text'], padding=True, return_tensors="pt")
                strat_ids = strat_inputs.input_ids.to(self.device, non_blocking=True)
                strat_mask = strat_inputs.attention_mask.to(self.device, non_blocking=True)

                with self.model_lock:
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        strat_logits = self.model(strat_ids, attention_mask=strat_mask).logits.float()

                with torch.no_grad():
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        ref_strat_logits = self.ref_model(strat_ids, attention_mask=strat_mask).logits.float()

                s_preds = strat_logits[:, -1, :]
                s_targets = torch.zeros_like(s_preds)

                for b, target_dict in enumerate(batch_data['strategy_targets']):
                    for act, prob in target_dict.items():
                        s_targets[b, self.action_tokens[act]] = prob

                s_loss = F.cross_entropy(s_preds, s_targets)

                shift_logits_train = strat_logits[..., :-1, :].contiguous()
                shift_logits_ref = ref_strat_logits[..., :-1, :].contiguous()
                shift_mask = strat_mask[..., 1:].contiguous().bool()

                flat_logits_train = shift_logits_train.view(-1, shift_logits_train.size(-1))[shift_mask.view(-1)]
                flat_logits_ref = shift_logits_ref.view(-1, shift_logits_ref.size(-1))[shift_mask.view(-1)]

                kl_loss = F.kl_div(
                    F.log_softmax(flat_logits_train, dim=-1),
                    F.softmax(flat_logits_ref, dim=-1),
                    reduction='batchmean'
                )

                # Token-Space Regret
                reg_inputs = self.tokenizer(batch_data['regret_text'], padding=True, return_tensors="pt")
                reg_ids = reg_inputs.input_ids.to(self.device, non_blocking=True)
                reg_mask = reg_inputs.attention_mask.to(self.device, non_blocking=True)
                reg_mask[:, -1] = 1

                with self.model_lock:
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        reg_logits = self.model(reg_ids, attention_mask=reg_mask).logits.float()

                r_preds = reg_logits[:, -1, :]

                target_classes = [
                    (torch.abs(self.torch_sizes_float - max(1.0, float(reg)))).argmin().item()
                    for reg in batch_data['regret_targets']
                ]
                target_tensor = torch.tensor(target_classes, device=self.device, dtype=torch.long)

                start_idx = self.min_size_token_id
                end_idx = start_idx + len(self.sizes)
                b_token_preds = r_preds[:, start_idx:end_idx]

                probs = F.softmax(b_token_preds, dim=-1)

                # Compute predicted Cumulative Distribution Function (CDF)
                cdf_pred = torch.cumsum(probs, dim=-1)

                # Compute target CDF (Step function: 0 before the true class, 1 at and after)
                batch_size, num_classes = b_token_preds.shape
                indices = torch.arange(num_classes, device=self.device).expand(batch_size, -1)
                cdf_target = (indices >= target_tensor.unsqueeze(1)).float()

                # Calculate the distances between your specific token bins
                bin_distances = torch.diff(self.torch_sizes_float)  # Shape: [num_classes - 1]

                # Weighted 1D Wasserstein Loss
                # We slice [:, :-1] because the total mass must sum to 1,
                # making the difference at the final bin always exactly 0.
                cdf_diff = torch.abs(cdf_pred[:, :-1] - cdf_target[:, :-1])

                # Multiply the mass moved by the distance it had to travel, then average over batch
                r_loss = torch.mean(torch.sum(cdf_diff * bin_distances, dim=-1))

                # Optimization
                total_loss = s_loss + r_loss + (beta * kl_loss)

                if total_loss.requires_grad:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                del strat_logits, ref_strat_logits, reg_logits, strat_ids, reg_ids, strat_inputs, reg_inputs
                del shift_logits_train, shift_logits_ref, flat_logits_train, flat_logits_ref, batch_data

                self.global_updates += 1
                if self.global_updates % 10 == 0:
                    self.csv_writer.writerow([self.global_updates, total_loss.item(), s_loss.item(), r_loss.item(), self.batch_queue.qsize()])
                    self.log_file.flush()

                if self.global_updates % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"Upd {self.global_updates} | Total: {total_loss.item():.4f} (Strat: {s_loss.item():.4f}, Reg: {r_loss.item():.4f}, KL: {kl_loss.item():.4f})")

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
                        sim.call()  # Fallback for call-is-allin
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
                current_holder[act] = [finished_payoffs[k][plyr] for k in range(start, start + count)]

        if current_holder is not None:
            final_output.append(current_holder)

        for i, hand in enumerate(hand_list):
            action_space = hand.get_action_space()

            if 'min_bet' in final_output[i]:
                final_output[i]['raise'] = final_output[i].pop('min_bet')
            if 'max_bet' in final_output[i]:
                final_output[i]['allin'] = final_output[i].pop('max_bet')

            # Short-stack Call-is-Allin logic (Use pop to remove 'call' from targets)
            if 'min_bet' not in action_space and 'call' in final_output[i]:
                final_output[i]['allin'] = final_output[i].pop('call')

        return final_output

    @torch.inference_mode()
    def select_raise_batch(self, hands):
        if not hands: return []
        thread_encoders = self.get_thread_encoders(len(hands))
        results = list(self.executor.map(self.process_hand_cpu, zip(hands, thread_encoders[:len(hands)])))
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, _, _, _, _, actor_indices = zip(*results)
        tokenizer = self.get_thread_tokenizer()
        inputs = tokenizer(list(encoded_strs), padding=True, return_tensors="pt")

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

        thread_encoders = self.get_thread_encoders(len(hands))
        results = list(self.executor.map(self.process_hand_cpu, zip(hands, thread_encoders[:len(hands)])))

        # Unpack the hand state
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices = zip(
            *results)

        tokenizer = self.get_thread_tokenizer()

        flat_queries = []
        action_maps = []  # Tracks which legal actions belong to which hand index

        # Build the specific action queries based on legal moves
        for i in range(len(hands)):
            base_encoded = f"{encoded_strs[i]}<herop{actor_indices[i]}>"
            legal_actions = []

            if can_check[i]:
                legal_actions.append('check')
            else:
                legal_actions.append('fold')

            if can_call[i] and not call_is_allin[i]: legal_actions.append('call')
            if can_raise[i]: legal_actions.append('raise')
            if max_bets[i] > 0 or call_is_allin[i]: legal_actions.append('allin')

            action_maps.append(legal_actions)
            for act in legal_actions:
                # Append <unk> to query the Pain/Regret distribution head
                flat_queries.append(f"{base_encoded}<{act}><unk>")

        # Batched Forward Pass to get Pain Distributions
        inputs = tokenizer(flat_queries, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        logits = self.model(input_ids, attention_mask=attention_mask).logits[:, -1, :]

        # Calculate Expected Pain from the CDF bins
        start_id = self.min_size_token.item()
        num_sizes = len(self.sizes_floats)
        pain_logits = logits[:, start_id: start_id + num_sizes]

        # Convert logits to probabilities over the bins
        pain_probs = F.softmax(pain_logits, dim=-1)

        # Multiply by the actual bin values (1.0 to 500.0) and sum to get the continuous expected pain
        expected_pains = torch.sum(pain_probs * self.torch_sizes_float, dim=-1)  # Shape: [len(flat_queries)]

        final_actions = [''] * len(hands)
        final_sizes = [0] * len(hands)
        raise_indexes = []

        query_idx = 0

        # Strict Action Selection (Minimize Pain)
        for i, legal_actions in enumerate(action_maps):
            # Extract the pain values specific to this hand's legal actions
            hand_pains = expected_pains[query_idx: query_idx + len(legal_actions)]
            query_idx += len(legal_actions)

            # STRICT MINIMIZATION: Find the index of the action with the lowest expected pain
            best_action_idx = torch.argmin(hand_pains).item()
            chosen_act = legal_actions[best_action_idx]

            final_actions[i] = chosen_act

            if chosen_act == 'allin':
                final_sizes[i] = max_bets[i]
            elif chosen_act == 'raise':
                raise_indexes.append(i)

        del inputs, input_ids, attention_mask, logits, pain_logits, pain_probs, expected_pains

        # If any hand chose to raise, query the model again specifically for the sizing distribution
        if raise_indexes:
            r_encoded = [encoded_strs[i] for i in raise_indexes]
            r_inputs = tokenizer(r_encoded, padding=True, return_tensors="pt")
            r_ids = r_inputs.input_ids.to(self.device, non_blocking=True)
            r_mask = r_inputs.attention_mask.to(self.device, non_blocking=True)
            r_actors = self.hero_token_ids[list(np.array(actor_indices)[raise_indexes])].to(self.device)
            r_raise_tok = torch.full((len(raise_indexes), 1), self.raise_token_id, device=self.device)

            # Note: We do NOT append <unk> here, so the model knows to output the bet sizing distribution
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

    def process_hand_cpu(self, args):
        hand, encoder = args
        action_space = hand.get_action_space()
        pot_size = hand.pot_size()

        if 'min_bet' in action_space:
            idx = np.searchsorted(self.sizes_floats, (action_space['min_bet'] / pot_size) * 100, side='left')
            min_bet_token = self.min_size_token + min(idx, len(self.sizes_floats) - 1)
            max_bet = action_space['max_bet']
        else:
            min_bet_token = 0
            max_bet = 0

        turn_idx = hand.state.turn_index

        can_check = 'check' in action_space
        can_raise = 'min_bet' in action_space
        can_call = 'call' in action_space

        # If they can call but do not have enough chips to raise, calling puts them all-in
        call_is_allin = can_call and not can_raise

        return (
            encoder.encode(json.dumps(hand.get_u_hand(turn_idx))),
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
