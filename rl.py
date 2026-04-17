import sys
import os
import copy
import json
import random
from random import randint
import traceback
import numpy as np
import gc
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict
import threading
import queue
import time

sys.modules["markupsafe._speedups"] = None
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import schedulefree
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from sim_hand import Hand
from sim_encoder import Encoder

class GameTheoryRegretHead(torch.nn.Module):
    def __init__(self, hidden_size, num_actions=5):
        super().__init__()
        self.layer_norm1 = torch.nn.LayerNorm(hidden_size)
        self.dense1 = torch.nn.Linear(hidden_size, hidden_size)
        self.act1 = torch.nn.GELU()

        self.layer_norm2 = torch.nn.LayerNorm(hidden_size)
        self.dense2 = torch.nn.Linear(hidden_size, hidden_size)
        self.act2 = torch.nn.GELU()

        self.out_proj = torch.nn.Linear(hidden_size, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        gain = 2 ** 0.5
        torch.nn.init.orthogonal_(self.dense1.weight, gain=gain)
        torch.nn.init.zeros_(self.dense1.bias)
        torch.nn.init.orthogonal_(self.dense2.weight, gain=gain)
        torch.nn.init.zeros_(self.dense2.bias)
        torch.nn.init.orthogonal_(self.out_proj.weight, gain=0.01)
        torch.nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states):
        x = self.layer_norm1(hidden_states)
        x = self.dense1(x)
        x = self.act1(x)

        res = x
        x = self.layer_norm2(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = x + res

        return self.out_proj(x)


class PokerDualHeadModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_config(config)
        self.regret_head = GameTheoryRegretHead(config.hidden_size, num_actions=5)

    def load_regret_head_weights(self, path, device):
        self.regret_head.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def load_base_weights(self, path, device):
        self.lm.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        base_transformer = getattr(self.lm, self.lm.base_model_prefix)
        base_outputs = base_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True
        )
        last_hidden_state = base_outputs.last_hidden_state
        logits = self.lm.lm_head(last_hidden_state)
        ev_preds = self.regret_head(last_hidden_state)

        outputs = CausalLMOutputWithPast(logits=logits, past_key_values=base_outputs.past_key_values)
        return outputs, ev_preds


class ThreadSafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def get(self):
        with self.lock: return self.value

    def increment(self):
        with self.lock: self.value += 1


class Simulator:
    def __init__(self):
        self.device = 'cuda'
        torch.set_float32_matmul_precision('high')
        self.thread_local = threading.local()

        config = AutoConfig.from_pretrained('./config.json')

        self.model = PokerDualHeadModel(config).to(self.device)
        self.model.load_base_weights('GEN-17600000.pt', self.device)
        self.model.share_memory()

        self.ref_model = PokerDualHeadModel(config).to(self.device)
        self.ref_model.load_base_weights('GEN-17600000.pt', self.device)
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.ref_model.share_memory()

        base_tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        base_tokenizer.padding_side = "left"
        base_tokenizer.pad_token = base_tokenizer.unk_token
        self.unk_token_id = base_tokenizer.unk_token_id

        self.action_names = ['fold', 'check', 'call', 'raise', 'allin']
        self.action_to_idx = {name: i for i, name in enumerate(self.action_names)}
        self.action_tokens = {name: base_tokenizer.encode(f"<{name}>")[0] for name in self.action_names}

        self.num_classes = 700
        self.min_bb_val = -200
        self.class_values = torch.arange(self.min_bb_val, self.min_bb_val + self.num_classes, device=self.device).float()

        self.fold_token_id = base_tokenizer.encode("<xxx>")[0]
        self.check_token_id = base_tokenizer.encode("<xxx>")[0]
        self.call_token_id = base_tokenizer.encode("<xxx>")[0]
        self.raise_token_id = base_tokenizer.encode("<xxx>")[0]
        self.allin_token_id = base_tokenizer.encode("<xxx>")[0]
        self.min_size_token_id = base_tokenizer.encode("<xxx>")[0]
        self.min_size_token = torch.tensor([self.min_size_token_id]).to(self.device)

        self.sizes = np.array(list(range(1, 5)), dtype=np.float32)
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=1e-6,
            warmup_steps=0,
            betas=(0.9, 0.999),
            weight_decay=0.0
        )
        self.optimizer.train()

        self.n_sims = 16
        self.gen_batch_size = 8
        self.train_batch_size = 32
        self.global_updates = 0
        self.inference_queue = queue.Queue()

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
            self.thread_local.encoders = [Encoder() for _ in range(self.gen_batch_size)]
        return self.thread_local.encoders

    @property
    def tokenizers(self):
        if not hasattr(self.thread_local, 'tokenizers'):
            toks = []
            for _ in range(self.gen_batch_size):
                tok = AutoTokenizer.from_pretrained('./opt-it-2')
                tok.padding_side = "left"
                tok.pad_token = tok.unk_token
                toks.append(tok)
            self.thread_local.tokenizers = toks
        return self.thread_local.tokenizers

    def inference_server_loop(self):
        print("GPU Inference Server Thread started (Zero-Delay Batching).")
        self.model.eval()
        while True:
            try:
                req = self.inference_queue.get()
                reqs = [req]
                total_queries = len(req['queries'])
                target_batch_size = 256

                while total_queries < target_batch_size:
                    try:
                        new_req = self.inference_queue.get_nowait()
                        reqs.append(new_req)
                        total_queries += len(new_req['queries'])
                    except queue.Empty:
                        break

                action_reqs = [r for r in reqs if r['type'] == 'action']
                raise_reqs = [r for r in reqs if r['type'] == 'raise']

                all_queries = []
                action_slices = []
                raise_slices = []

                current_pos = 0
                for r in action_reqs:
                    n = len(r['queries'])
                    all_queries.extend(r['queries'])
                    action_slices.append((current_pos, current_pos + n))
                    current_pos += n

                for r in raise_reqs:
                    n = len(r['queries'])
                    all_queries.extend(r['queries'])
                    raise_slices.append((current_pos, current_pos + n))
                    current_pos += n

                if not all_queries:
                    continue

                inputs = self.tokenizer(all_queries, padding=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device, non_blocking=True)
                attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

                with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs, regret_logits = self.model(input_ids, attention_mask=attention_mask)
                    action_results = regret_logits[:, -1, :].float()
                    raise_results = outputs.logits[:, -1, :].float()

                for idx, r in enumerate(action_reqs):
                    s, e = action_slices[idx]
                    r['result']['data'] = action_results[s:e]
                    r['event'].set()

                for idx, r in enumerate(raise_reqs):
                    s, e = raise_slices[idx]
                    r['result']['data'] = raise_results[s:e]
                    r['event'].set()

            except Exception as e:
                print(f"Inference Server Error: {e}")
                traceback.print_exc()

    def generate_experience(self, current_train_count):
        hands, hero_ev_data, hero_indices, full_seqs = self.batch_generate_hands(self.gen_batch_size)
        experiences = []

        for i, hand in enumerate(hands):
            if not hero_ev_data[i]: continue

            big_blind = float(hand.big_blind) if hand.big_blind > 0 else 1.0
            hand_decision_counts = []
            hand_legal_actions = []
            hand_ev_scores = []

            for node_data in hero_ev_data[i]:
                ev_dict = node_data['evs']
                samples_bb = {
                    act_str: [float(s) / big_blind for s in samples]
                    for act_str, samples in ev_dict.items()
                    if act_str in self.action_tokens and samples
                }

                if not samples_bb: continue

                acts = list(samples_bb.keys())
                hand_decision_counts.append(node_data['decision_token_count'])
                hand_legal_actions.append(acts)
                hand_ev_scores.append(samples_bb)

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
        self.model.eval()
        gc.disable()

        while True:
            try:
                current_tc = train_count_obj.get()
                with torch.no_grad():
                    experiences = self.generate_experience(current_tc)

                for exp in experiences:
                    q.put(exp)
                gc.collect()

            except Exception as e:
                print(f"Generator {worker_id} Exception: {e}")
                traceback.print_exc()

    def run_trainer(self, q, train_count_obj):
        print("Starting Central Trainer Loop (Dual-Head Network)...")

        log_file = open('training_logs.csv', mode='w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Update', 'Policy_Loss', 'Regret_Loss', 'KL_Loss'])

        batch_memory = []
        policy_losses = []
        regret_losses = []

        while self.global_updates < 1000000:
            with global_profiler.profile("Trainer: Waiting for Queue (GPU IDLE)"):
                while len(batch_memory) < self.train_batch_size:
                    try:
                        exp = q.get(timeout=1)
                        if exp['train_count'] == train_count_obj.get():
                            batch_memory.append(exp)
                    except queue.Empty:
                        continue

            batch_samples = batch_memory[:self.train_batch_size]
            batch_memory = batch_memory[self.train_batch_size:]

            full_texts = [s['full_text'] for s in batch_samples]
            inputs = self.tokenizer(full_texts, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device, non_blocking=True)
            attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

            seq_len = input_ids.size(1)
            unpadded_lengths_cpu = attention_mask.sum(dim=1).long().tolist()

            self.model.train()
            with global_profiler.profile("Trainer: GPU Forward Pass"):
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs, regret_logits = self.model(input_ids, attention_mask=attention_mask)
                active_logits = outputs.logits.float()
                regret_logits = regret_logits.float()

                with torch.no_grad():
                    with global_profiler.profile("Trainer: CPU Loss Calculation"):
                        ref_outputs, _ = self.ref_model(input_ids, attention_mask=attention_mask)
                        ref_logits = ref_outputs.logits.float()

                batch_indices = []
                seq_indices = []
                target_evs_list = []
                legal_masks_list = []
                dynamic_weights = []

                kl_mask = attention_mask.clone().float()

                for b in range(self.train_batch_size):
                    counts = batch_samples[b]['decision_token_counts']
                    pad_offset = seq_len - unpadded_lengths_cpu[b]

                    for i, count in enumerate(counts):
                        hero_token_idx = pad_offset + count
                        batch_indices.append(b)
                        seq_indices.append(hero_token_idx)

                        kl_mask[b, hero_token_idx] = 0.0

                        acts = batch_samples[b]['legal_actions'][i]
                        samples_dict = batch_samples[b]['ev_scores'][i]

                        t_evs = [0.0] * 5
                        l_mask = [False] * 5
                        act_means = []

                        for act_str in acts:
                            act_idx = self.action_to_idx[act_str]
                            mean_val = float(np.mean(samples_dict[act_str]))
                            act_means.append(mean_val)
                            t_evs[act_idx] = mean_val / 100.0
                            l_mask[act_idx] = True

                        target_evs_list.append(t_evs)
                        legal_masks_list.append(l_mask)

                        spread = max(act_means) - min(act_means) if act_means else 0.0
                        dynamic_weights.append(float(spread) + 1e-3)

                total_nodes = len(batch_indices)

                if total_nodes > 0:
                    b_idx = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
                    s_idx = torch.tensor(seq_indices, device=self.device, dtype=torch.long)
                    t_evs_tensor = torch.tensor(target_evs_list, device=self.device, dtype=torch.float32)
                    l_mask_tensor = torch.tensor(legal_masks_list, device=self.device, dtype=torch.bool)

                    weights_tensor = torch.tensor(dynamic_weights, device=self.device, dtype=torch.float32)
                    weights_tensor = weights_tensor / weights_tensor.mean()

                    node_ev_preds = regret_logits[b_idx, s_idx, :]
                    hero_logits = active_logits[b_idx, s_idx, :]

                    mse_all = self.mse_loss(node_ev_preds, t_evs_tensor)
                    masked_mse = mse_all.masked_fill(~l_mask_tensor, 0.0)
                    node_regret_losses = masked_mse.sum(dim=-1) / l_mask_tensor.sum(dim=-1).float()

                    masked_targets = t_evs_tensor.masked_fill(~l_mask_tensor, 0.0)
                    legal_counts = l_mask_tensor.sum(dim=-1, keepdim=True).float()
                    mean_target_evs = masked_targets.sum(dim=-1, keepdim=True) / legal_counts

                    target_regrets = t_evs_tensor - mean_target_evs
                    pos_regrets = torch.clamp(target_regrets, min=0.0).masked_fill(~l_mask_tensor, 0.0)
                    sum_pos = pos_regrets.sum(dim=-1, keepdim=True)

                    safe_sum = torch.clamp(sum_pos, min=1e-8)
                    uniform_probs = l_mask_tensor.float() / legal_counts
                    target_probs_5 = torch.where(sum_pos > 0, pos_regrets / safe_sum, uniform_probs)

                    vocab_targets = torch.zeros_like(hero_logits)
                    for act_str, act_idx in self.action_to_idx.items():
                        vocab_idx = self.action_tokens[act_str]
                        vocab_targets[:, vocab_idx] = target_probs_5[:, act_idx]

                    node_policy_losses = F.cross_entropy(hero_logits, vocab_targets, reduction='none')

                    regret_loss = torch.mean(node_regret_losses * weights_tensor)
                    policy_loss = torch.mean(node_policy_losses * weights_tensor)

                else:
                    regret_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            log_probs_train = F.log_softmax(active_logits, dim=-1)
            probs_ref = F.softmax(ref_logits, dim=-1)
            kl_unreduced = F.kl_div(log_probs_train, probs_ref, reduction='none').sum(dim=-1)

            masked_kl = kl_unreduced * kl_mask
            active_kl_tokens = torch.clamp(kl_mask.sum(), min=1.0)
            kl_loss = 100 * masked_kl.sum() / active_kl_tokens

            if self.global_updates < 100000:
                policy_mask = (self.global_updates - 90000) / 10000 if self.global_updates > 90000 else 0.0
            else:
                policy_mask = 1.0

            total_loss = policy_mask * policy_loss + regret_loss + kl_loss
            regret_losses.append(regret_loss.item())
            policy_losses.append(policy_loss.item())

            with global_profiler.profile("Trainer: GPU Backward & Step"):
                if total_loss.requires_grad:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.global_updates += 1

            if self.global_updates % 50 == 0:
                global_profiler.report_and_reset(self.global_updates)
                csv_writer.writerow([self.global_updates, policy_loss.item(), regret_loss.item(), kl_loss.item()])
                log_file.flush()

            if self.global_updates % 5000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                mean_policy_losses = np.mean(policy_losses)
                mean_regret_losses = np.mean(regret_losses)
                print(f"Upd {self.global_updates} | Nodes: {total_nodes} | Pol: {mean_policy_losses:.4f} | Reg: {mean_regret_losses:.4f} | KL: {kl_loss.item():.4f}")
                regret_losses = []
                policy_losses = []
                torch.save(self.model.lm.state_dict(), f"RL-{self.global_updates}.pt")
                torch.save(self.model.regret_head.state_dict(), f"RL-{self.global_updates}-RegretHead.pt")

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
                with global_profiler.profile("Worker: MCTS Generate Bulk EVs (Total)"):
                    with global_profiler.profile("Worker: GPU Raise Sizing"):
                        cf_sizes = self.select_raise_batch(ev_hand_objs)

                    with global_profiler.profile("Worker: CPU Deepcopy"):
                        temp_hands = []
                        valid_actions_list = []
                        for i, (hand, sz) in enumerate(zip(ev_hand_objs, cf_sizes)):
                            action_space = hand.get_action_space()
                            max_bet = action_space.get('max_bet', 0)
                            valid_actions = {'fold', 'check', 'call', 'min_bet', 'max_bet'}
                            
                            can_raise = 'min_bet' in action_space
                            if can_raise and action_space['min_bet'] >= max_bet:
                                can_raise = False
                            
                            can_call = 'call' in action_space
                            call_is_allin = can_call and not can_raise
                            
                            if not can_raise: valid_actions.discard('min_bet')
                            if call_is_allin: valid_actions.discard('max_bet')
                            
                            if max_bet > 0 and sz > 0 and max_bet > 4 * sz:
                                valid_actions.discard('max_bet')
                            elif max_bet > 0 and sz > 0 and sz >= (0.5 * max_bet):
                                valid_actions.discard('min_bet')
                            
                            valid_actions_list.append(valid_actions)
                            temp_hands.append(copy.deepcopy(hand))
                        
                    bulk_results = self.generate_bulk_evs_instrumented(ev_hand_objs, cf_sizes, temp_hands, valid_actions_list)

                with global_profiler.profile("Worker: CPU JSON Serialization"):
                    for k, global_idx in enumerate(ev_indices):
                        current_state_json = json.dumps(ev_hand_objs[k].get_u_hand(hero_indices[global_idx]))
                        hero_ev_data[global_idx].append({'state_json': current_state_json, 'evs': bulk_results[k]})

            with global_profiler.profile("Worker: GPU Action Selection"):
                actions, sizes = self.select_action_batch(current_hands)
            next_active_indices = []

            with global_profiler.profile("Worker: Apply Action to Main Hand"):
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
                        if size > 0 and size >= (0.5 * action_space.get('max_bet', 0)):
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
            current_tokenizers = self.tokenizers

            def _process_entire_trajectory(encoder, hand, ev_nodes, hero_idx, thread_tokenizer):
                final_seq = encoder.encode(json.dumps(hand.get_u_hand(hero_idx)), True, True)
                valid_nodes = []
                for node in ev_nodes:
                    state_json = node['state_json']
                    hist_encoded = encoder.encode(state_json)
                    prefix_str = f"{hist_encoded}<herop{hero_idx}>"
                    prefix_ids = thread_tokenizer.encode(prefix_str, add_special_tokens=True)
                    exact_token_idx = len(prefix_ids) - 1

                    valid_nodes.append({
                        'decision_token_count': exact_token_idx,
                        'evs': node['evs']
                    })
                return final_seq, valid_nodes

            with global_profiler.profile("Worker: ThreadPool Encoding"):
                with ThreadPoolExecutor(max_workers=len(valid_indices)) as executor:
                    futures_list = [
                        executor.submit(
                            _process_entire_trajectory,
                            current_encoders[idx],
                            hands[idx],
                            hero_ev_data[idx],
                            hero_indices[idx],
                            current_tokenizers[idx]
                        )
                        for idx in valid_indices
                    ]

                    for idx, future in zip(valid_indices, futures_list):
                        final_seq, valid_nodes = future.result()
                        full_seqs[idx] = final_seq
                        hero_ev_data[idx] = valid_nodes

        return hands, hero_ev_data, hero_indices, full_seqs

    @torch.no_grad()
    def generate_bulk_evs_instrumented(self, hand_list, size_list, temp_hands, valid_actions_list):
        all_sims = []
        registry = []
        
        with global_profiler.profile("Worker: Setup Initial MCTS Nodes"):
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
                valid_actions = valid_actions_list[i]
    
                for root_action in action_space.keys():
                    if root_action not in valid_actions or root_action == 'fold': continue
    
                    with global_profiler.profile("Worker: Initial MCTS Actions"):
                        temp_hand = temp_hands[i]
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
                        with global_profiler.profile("Worker: CPU Deepcopy inside MCTS setup"):
                            for _ in range(self.n_sims):
                                sim_clone = copy.deepcopy(temp_hand)
                                sim_clone.shuffle()
                                all_sims.append(sim_clone)
                        registry.append({'start': start_idx, 'count': self.n_sims, 'player': player, 'action': root_action,
                                         'is_calc': True})
                        res[root_action] = "PENDING"
                        
                    temp_hands[i] = copy.deepcopy(hand)

        active_sim_indices = list(range(len(all_sims)))
        finished_payoffs = [None] * len(all_sims)
        
        sim_depth_counter = 0

        with global_profiler.profile("Worker: Complete Rollout Loop"):
            while active_sim_indices:
                sim_depth_counter += 1
                current_sim_batch = [all_sims[k] for k in active_sim_indices]
                
                with global_profiler.profile("Worker: Rollout Action Selection (GPU)"):
                    actions, sizes = self.select_action_batch(current_sim_batch)
                
                global_profiler.record_mcts_step(len(active_sim_indices), actions)
                    
                next_active = []
                terminals_reached = 0

                with global_profiler.profile("Worker: Rollout Apply Action"):
                    for j, (act, sz) in enumerate(zip(actions, sizes)):
                        sim_idx = active_sim_indices[j]
                        sim = all_sims[sim_idx]
                        
                        with global_profiler.profile(f"Rollout Action: {act}"):
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
                                if sz > 0 and sz >= (0.5 * action_space.get('max_bet', 0)):
                                    sim.bet_or_raise(action_space.get('max_bet', 0))
                                else:
                                    sim.bet_or_raise(sz)
        
                        with global_profiler.profile("Rollout Eval: Check Terminal Payoffs"):
                            if sim.done:
                                p = sim.state.payoffs
                                for x in range(len(p)):
                                    if p[x] > 0: p[x] -= min(p[x] * 0.05, 2 * sim.big_blind)
                                finished_payoffs[sim_idx] = p
                                terminals_reached += 1
                            else:
                                next_active.append(sim_idx)
                                
                active_sim_indices = next_active
                global_profiler.record_mcts_terminal(terminals_reached)

        global_profiler.record_rollout_depth(sim_depth_counter)

        final_output = []
        current_holder = None

        with global_profiler.profile("Worker: Aggregate MCTS Results"):
            for item in registry:
                if not item['is_calc']:
                    if current_holder is not None: final_output.append(current_holder)
                    current_holder = item['holder']
                else:
                    start, count, plyr, act = item['start'], item['count'], item['player'], item['action']
                    current_holder[act] = [finished_payoffs[k][plyr] for k in range(start, start + count)]
    
            if current_holder is not None: final_output.append(current_holder)
    
            for i, hand in enumerate(hand_list):
                action_space = hand.get_action_space()
                max_bet = action_space.get('max_bet', 0)
                can_raise = 'min_bet' in action_space
                if can_raise and action_space['min_bet'] >= max_bet: can_raise = False
                can_call = 'call' in action_space
                call_is_allin = can_call and not can_raise
    
                if 'min_bet' in final_output[i]: final_output[i]['raise'] = final_output[i].pop('min_bet')
                if 'max_bet' in final_output[i]: final_output[i]['allin'] = final_output[i].pop('max_bet')
                if call_is_allin and 'call' in final_output[i]: final_output[i]['allin'] = final_output[i].pop('call')

        return final_output

    @torch.no_grad()
    def select_raise_batch(self, hands):
        if not hands: return []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, _, _, _, _, actor_indices = zip(*results)

        queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}><raise>" for i in range(len(hands))]

        evt = threading.Event()
        res_container = {}
        self.inference_queue.put({
            'type': 'raise',
            'queries': queries,
            'event': evt,
            'result': res_container
        })
        evt.wait()

        last_logits = res_container['data']
        start_id = self.min_size_token_id
        num_sizes = len(self.sizes_floats)

        size_logits = last_logits[:, start_id: start_id + num_sizes]

        batch_min_tokens = torch.tensor(min_bet_tokens, device=self.device)
        offsets = (batch_min_tokens - start_id).unsqueeze(1)
        size_indices = torch.arange(num_sizes, device=self.device).unsqueeze(0)

        size_logits = size_logits.masked_fill(~(size_indices >= offsets), float('-inf'))
        probs = torch.softmax(size_logits, dim=1)
        probs = torch.nan_to_num(probs, nan=1e-5)

        relative_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        chosen_percents = self.torch_sizes_float[relative_indices]

        bets = (torch.tensor(pot_sizes, device=self.device) * (chosen_percents / 100.0)).long()

        final_bets = [
            int(min(amt, cap) if cap > 0 else amt)
            for amt, cap in zip(bets.tolist(), max_bets)
        ]

        return final_bets

    @torch.no_grad()
    def select_action_batch(self, hands):
        if not hands: return [], []

        with global_profiler.profile("Action Select: Process Hand CPU"):
            results = [self.process_hand_cpu(hand) for hand in hands]
            
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices = zip(*results)

        pre_sampled_sizes = self.select_raise_batch(hands) if hands else []

        queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}>" for i in range(len(hands))]

        evt = threading.Event()
        res_container = {}
        
        with global_profiler.profile("Action Select: Inference Queue Wait"):
            self.inference_queue.put({
                'type': 'action',
                'queries': queries,
                'event': evt,
                'result': res_container
            })
            evt.wait()

        hero_ev_preds = res_container['data']

        final_actions = [''] * len(hands)
        final_sizes = [0] * len(hands)

        with global_profiler.profile("Action Select: Output Processing"):
            for i in range(len(hands)):
                legal_acts = []
                if not can_check[i]: legal_acts.append('fold')
                if can_check[i]: legal_acts.append('check')
                if can_call[i] and not call_is_allin[i]: legal_acts.append('call')
    
                can_raise_flag = can_raise[i]
                can_allin_flag = max_bets[i] > 0 or call_is_allin[i]
                raise_sz = pre_sampled_sizes[i]

                if max_bets[i] > 0 and raise_sz > 0 and max_bets[i] > 4 * raise_sz:
                    can_allin_flag = False
                elif max_bets[i] > 0 and raise_sz > 0 and raise_sz >= (0.5 * max_bets[i]):
                    can_raise_flag = False

                if can_raise_flag: legal_acts.append('raise')
                if can_allin_flag: legal_acts.append('allin')
    
                if not legal_acts: legal_acts = ['fold']
    
                act_indices = [self.action_to_idx[act] for act in legal_acts]
                expected_evs = hero_ev_preds[i, act_indices]
                temperature = 1.0
                stable_evs = expected_evs - expected_evs.max()
                probs = torch.softmax(stable_evs / temperature, dim=-1)
                probs = torch.clamp(probs, min=1e-5)
                probs = probs / probs.sum()

                chosen_idx = torch.multinomial(probs, num_samples=1).item()
                chosen_act = legal_acts[chosen_idx]
    
                final_actions[i] = chosen_act
    
                if chosen_act == 'allin':
                    final_sizes[i] = max_bets[i]
                elif chosen_act == 'raise':
                    final_sizes[i] = pre_sampled_sizes[i]

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

    experience_queue = queue.Queue(maxsize=10000)
    global_train_count = ThreadSafeCounter()

    server_thread = threading.Thread(target=sim.inference_server_loop, daemon=True)
    server_thread.start()

    num_generators = 16
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
