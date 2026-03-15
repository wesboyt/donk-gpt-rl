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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DynamicCache

from sim_hand import Hand
from sim_encoder import Encoder
import threading
import queue
import concurrent.futures


class NonUniformWasserstein1DLoss(torch.nn.Module):
    def __init__(self, sizes_tensor):
        super().__init__()
        # Calculate the physical distance between consecutive atoms
        # sizes_tensor shape: [num_sizes]
        # delta_z shape: [num_sizes - 1]
        self.delta_z = torch.diff(sizes_tensor)

    def forward(self, pred_logits, target_probs):
        pred_probs = torch.softmax(pred_logits, dim=-1)

        # Calculate Cumulative Distribution Functions
        # We drop the last element because diffs only exist between elements
        pred_cdf = torch.cumsum(pred_probs, dim=-1)[..., :-1]
        target_cdf = torch.cumsum(target_probs, dim=-1)[..., :-1]

        cdf_diff = torch.abs(pred_cdf - target_cdf)

        # Multiply the CDF mass differences by the physical step sizes and sum
        return torch.sum(cdf_diff * self.delta_z, dim=-1).mean()


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

        # Training Model
        self.model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.model.share_memory()  # Explicitly share across processes

        self.ref_model = AutoModelForCausalLM.from_config(config).to(self.device)
        self.ref_model.load_state_dict(torch.load('GEN-17600000.pt', map_location=self.device, weights_only=True))
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        self.ref_model.share_memory()


        base_tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        base_tokenizer.padding_side = "left"
        base_tokenizer.pad_token = base_tokenizer.unk_token
        self.unk_token_id = base_tokenizer.unk_token_id

        # Action Tokens
        self.fold_token_id = base_tokenizer.encode("<xxx>")[0]
        self.check_token_id = base_tokenizer.encode("<xxx>")[0]
        self.call_token_id = base_tokenizer.encode("<xxx>")[0]
        self.raise_token_id = base_tokenizer.encode("<xxx>")[0]
        self.allin_token_id = base_tokenizer.encode("<xxx>")[0]

        self.min_size_token_id = base_tokenizer.encode("<xxx>")[0]
        self.min_size_token = torch.tensor([self.min_size_token_id]).to(self.device)

        # Core Tokens mapping
        self.hero_token_ids = torch.tensor([base_tokenizer.encode(f"<xxx>")[0] for i in range(6)],
                                           device=self.device)
        self.action_tokens = {
            'fold': self.fold_token_id,
            'check': self.check_token_id,
            'call': self.call_token_id,
            'raise': self.raise_token_id,
            'allin': self.allin_token_id
        }

        # Token-Space Regret Supports
        self.sizes = np.array(list(range(1, 5))..., dtype=np.float32)
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()

        self.wasserstein_loss = NonUniformWasserstein1DLoss(self.torch_sizes_float)

        # Optimizer (Only used by trainer process)
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=1e-6,
            warmup_steps=0,
            betas=(0.9, 0.999),
            weight_decay=0.0  # <--- FIX: Disabled to prevent logit compression
        )
        self.optimizer.train()

        # Engine Params
        self.n_sims = 8
        self.batch_size = 8
        self.global_updates = 0



    @property
    def tokenizer(self):
        """Lazily instantiates and caches a Tokenizer specific to the calling thread."""
        if not hasattr(self.thread_local, 'tokenizer'):
            tok = AutoTokenizer.from_pretrained('./opt-it-2')
            tok.padding_side = "left"
            tok.pad_token = tok.unk_token
            self.thread_local.tokenizer = tok
        return self.thread_local.tokenizer

    @property
    def encoder(self):
        """Lazily instantiates and caches an Encoder specific to the calling thread."""
        if not hasattr(self.thread_local, 'encoder'):
            self.thread_local.encoder = Encoder()
        return self.thread_local.encoder

    @property
    def encoders(self):
        """Lazily instantiates and caches batch of encoders specific to the calling thread."""
        if not hasattr(self.thread_local, 'encoders'):
            encoders = []
            for i in range(self.batch_size):
                encoders.append(Encoder())
            self.thread_local.encoders = encoders
        return self.thread_local.encoders

    def generate_experience(self, current_train_count):
        """Generates a chunk of experiences tagged with the current train_count"""
        hands, hero_ev_data, hero_indices, full_seqs = self.batch_generate_hands(self.batch_size)
        experiences = []

        for i, hand in enumerate(hands):
            if not hero_ev_data[i]: continue

            big_blind = float(hand.big_blind) if hand.big_blind > 0 else 1.0

            hand_decision_counts = []
            hand_legal_actions = []
            hand_ev_scores = []

            # hero_ev_data[i] is now a pre-filtered list of dictionaries from the ThreadPool
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

                # Append the pre-calculated integer directly
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
        """Generator thread loop that continuously fills the queue."""
        print(f"Generator Thread {worker_id} started.")
        self.model.eval()

        while True:
            try:
                current_tc = train_count_obj.get()

                with torch.no_grad():
                    experiences = self.generate_experience(current_tc)

                for exp in experiences:
                    # Blocks if queue is full
                    q.put(exp)
            except Exception as e:
                print(f"Generator {worker_id} Exception: {e}")
                traceback.print_exc()

    def run_trainer(self, q, train_count_obj):
        """Main training loop with Bifurcated Regret Prediction."""
        print("Starting Central Trainer Loop (Double Neural CFR)...")

        log_file = open('training_logs.csv', mode='w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Update', 'Policy_Loss', 'Regret_Loss', 'KL_Loss'])

        batch_memory = []

        while self.global_updates < 40000:
            # 1. Drain Queue and Batch Preparation (Unchanged)
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

            self.model.train()
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
                active_logits = outputs.logits.float()

                detached_cache_data = []
                for layer in outputs.past_key_values.layers:
                    k = layer.keys.detach() if layer.keys is not None else None
                    v = layer.values.detach() if layer.values is not None else None
                    sliding_tensor = getattr(layer, "_sliding_window_tensor", None)

                    if sliding_tensor is not None:
                        detached_cache_data.append((k, v, sliding_tensor))
                    else:
                        detached_cache_data.append((k, v))

                detached_pkv = DynamicCache(detached_cache_data)
                if self.global_updates == 1:
                    first_k_tensor = detached_pkv.layers[0].keys
                    assert not first_k_tensor.requires_grad, "Error: Detached PKV still requires grad."
                    assert first_k_tensor.grad_fn is None, "Error: Detached PKV still has a grad_fn."

                with torch.no_grad():
                    ref_logits = self.ref_model(input_ids, attention_mask=attention_mask).logits.float()

            policy_loss = torch.tensor(0.0, device=self.device)
            regret_loss = torch.tensor(0.0, device=self.device)
            total_nodes = 0

            # --- 3. REGRET BRANCH & STRATEGY LABEL GENERATION ---
            for b in range(self.batch_size):
                counts = batch_samples[b]['decision_token_counts']
                for i, count in enumerate(counts):
                    pad_offset = seq_len - unpadded_lengths_cpu[b]
                    hero_token_idx = pad_offset + count - 1
                    total_nodes += 1

                    acts = batch_samples[b]['legal_actions'][i]
                    scores = batch_samples[b]['ev_scores'][i]

                    # 3A. Calculate Distance from Optimal (Shortfall)
                    max_score = max(scores)
                    true_distances = [max_score - s for s in scores]

                    num_acts = len(acts)
                    num_sizes = len(self.sizes_floats)

                    suffix_ids = torch.empty((num_acts, 2), dtype=torch.long, device=self.device)
                    target_bucket_probs = torch.zeros((num_acts, num_sizes), device=self.device)

                    for a_idx, act_str in enumerate(acts):
                        suffix_ids[a_idx, 0] = self.unk_token_id
                        suffix_ids[a_idx, 1] = self.action_tokens[act_str]

                        # 3B. Find the token index where size > distance
                        bucket_idx = num_sizes - 1  # Default to max size if it exceeds bounds
                        for j, size_val in enumerate(self.sizes_floats):
                            if size_val > true_distances[a_idx]:
                                bucket_idx = j
                                break

                        # Set a one-hot distribution at the exact target bucket
                        target_bucket_probs[a_idx, bucket_idx] = 1.0

                    # Expand the detached PKV specifically for this node's branch
                    node_cache_data = []
                    for layer in detached_pkv.layers:
                        k = layer.keys[b:b + 1, :, :hero_token_idx + 1, :].expand(num_acts, -1, -1, -1)
                        v = layer.values[b:b + 1, :, :hero_token_idx + 1, :].expand(num_acts, -1, -1, -1)
                        sliding_tensor = getattr(layer, "_sliding_window_tensor", None)

                        if sliding_tensor is not None:
                            node_cache_data.append((k, v, sliding_tensor))
                        else:
                            node_cache_data.append((k, v))

                    node_pkv = DynamicCache(node_cache_data)

                    branch_mask = torch.ones((num_acts, hero_token_idx + 3), device=self.device)

                    # Shallow Forward Pass for <unk><action>
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        branch_outputs = self.model(
                            input_ids=suffix_ids,
                            attention_mask=branch_mask,
                            past_key_values=node_pkv
                        )

                    branch_logits = branch_outputs.logits.float()

                    # -------------------------------------------------------------
                    # OPTIMAL TRANSPORT REGRET LOSS
                    # Slice out only the vocabulary logits corresponding to the sizes
                    # -------------------------------------------------------------
                    bucket_logits = branch_logits[:, 1, self.min_size_token_id: self.min_size_token_id + num_sizes]
                    regret_loss += self.wasserstein_loss(bucket_logits, target_bucket_probs)

                    # 3C. Generate Strategy Labels from Predictions
                    with torch.no_grad():
                        pred_bucket_probs = F.softmax(bucket_logits, dim=-1)

                        # Expected distance from optimal
                        expected_distances = torch.sum(pred_bucket_probs * self.torch_sizes_float, dim=-1)

                        # Convert distances back to standard regrets:
                        # Value = -Distance. Regret = Value - Mean(Value) = Mean(Distance) - Distance
                        mean_distance = expected_distances.mean()
                        pred_regrets = mean_distance - expected_distances

                        # Regret Matching
                        pos_regrets = torch.clamp(pred_regrets, min=0.0)
                        sum_pos_regrets = pos_regrets.sum()

                        target_policy = torch.zeros(active_logits.shape[-1], device=self.device)
                        if sum_pos_regrets > 0:
                            probs = pos_regrets / sum_pos_regrets
                        else:
                            probs = torch.ones(num_acts, device=self.device) / num_acts

                        act_token_ids = [self.action_tokens[a] for a in acts]
                        target_policy[act_token_ids] = probs

                    # Policy Loss (at the <hero> token)
                    hero_logits = active_logits[b, hero_token_idx, :]
                    policy_loss += F.cross_entropy(hero_logits, target_policy)

            # Normalize losses
            policy_loss = policy_loss / total_nodes
            regret_loss = regret_loss / total_nodes

            # ... end of batch loop ...

            policy_loss = policy_loss / total_nodes
            regret_loss = regret_loss / total_nodes

            # --- DIAGNOSTIC PART 2 ---
            if self.global_updates == 1:
                print("\n--- Running Gradient Leak Diagnostic ---")
                self.optimizer.zero_grad()

                # Backpropagate ONLY the regret branch
                regret_loss.backward(retain_graph=True)

                # Check the LM Head
                lm_head_weight = self.model.get_output_embeddings().weight
                if lm_head_weight.grad is not None and lm_head_weight.grad.abs().max().item() > 0.0:
                    print("[PASS] LM Head actively receiving regret gradients.")
                else:
                    print("[FAIL] LM Head is missing regret gradients.")

                # Check the Base Embeddings for Isolation
                embed_grad = self.model.get_input_embeddings().weight.grad
                if embed_grad is not None:
                    # Find all token IDs that received gradients
                    grad_mask = embed_grad.abs().sum(dim=1) > 0
                    updated_token_ids = torch.nonzero(grad_mask).squeeze().cpu().tolist()

                    if not isinstance(updated_token_ids, list):
                        updated_token_ids = [updated_token_ids]

                    expected_ids = {
                        self.unk_token_id,
                        self.fold_token_id, self.check_token_id,
                        self.call_token_id, self.raise_token_id,
                        self.allin_token_id
                    }

                    # Check if any token OUTSIDE the expected suffix tokens received a gradient
                    leak_ids = [tid for tid in updated_token_ids if tid not in expected_ids]

                    if len(leak_ids) == 0:
                        print(f"[PASS] Base embeddings isolated. Only suffix tokens updated: {updated_token_ids}")
                    else:
                        print(f"[FAIL] Gradient leak detected! Historical tokens modified: {leak_ids}")
                else:
                    print("[FAIL] No gradients found in base embeddings at all.")

                # Clear the diagnostic gradients
                self.optimizer.zero_grad()
                print("----------------------------------------\n")

            # --- 4. KL DIVERGENCE (Main Context Only) ---
            log_probs_train = F.log_softmax(active_logits, dim=-1)
            probs_ref = F.softmax(ref_logits, dim=-1)
            kl_unreduced = F.kl_div(log_probs_train, probs_ref, reduction='none').sum(dim=-1)
            masked_kl = kl_unreduced * attention_mask
            kl_loss = masked_kl.sum() / attention_mask.sum()

            # Combined Optimization
            total_loss = policy_loss + regret_loss + kl_loss

            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.global_updates += 1

            if self.global_updates % 10 == 0:
                csv_writer.writerow([self.global_updates, policy_loss.item(), regret_loss.item(), kl_loss.item()])
                log_file.flush()

            if self.global_updates % 500 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Upd {self.global_updates} | Nodes: {total_nodes} | Pol: {policy_loss.item():.4f} | Reg: {regret_loss.item():.4f} | KL: {kl_loss.item():.4f}")
                torch.save(self.model.state_dict(), f"RL-{self.global_updates}.pt")

    def batch_generate_hands(self, n_hands):  # STRIPPED parameter
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
                # STRIPPED: Removed 'and hand.state.street_index >= street_cutoff'
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
                    # Appends naturally, keeping track of all nodes
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

        # 2. Only spin up threads and run encodes for the valid hands
        # 2. Spin up threads to encode the final hands AND all historical nodes concurrently
        if valid_indices:
            current_encoders = self.encoders

            def _process_entire_trajectory(encoder, hand, ev_nodes, hero_idx):
                # Encode the final full sequence
                final_seq = encoder.encode(json.dumps(hand.get_u_hand(hero_idx)), True, True)

                # Parse and encode the historical nodes in parallel
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

                # Map results and directly overwrite the raw JSON strings with the processed node data
                for idx, future in zip(valid_indices, futures):
                    final_seq, valid_nodes = future.result()
                    full_seqs[idx] = final_seq
                    hero_ev_data[idx] = valid_nodes

        return hands, hero_ev_data, hero_indices, full_seqs

    @torch.no_grad()
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

            # Filter out all-in if it is more than 4x the sampled raise amount
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


    @torch.no_grad()
    def select_raise_batch(self, hands):
        if not hands: return []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, _, _, _, _, actor_indices = zip(*results)

        # Built-in String concatenation fix
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

    @torch.no_grad()
    def select_action_batch(self, hands):
        if not hands: return [], []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices = zip(*results)

        queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}>" for i in range(len(hands))]

        inputs = self.tokenizer(queries, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        # ---------------------------------------------------------
        # 1. BASE FORWARD PASS (Up to <hero>)
        # ---------------------------------------------------------
        outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
        pkv = outputs.past_key_values

        # ---------------------------------------------------------
        # 2. PREPARE LEGAL ACTIONS & VECTORIZED SUFFIXES
        # ---------------------------------------------------------
        flat_b_indices = []
        flat_act_ids = []
        hand_act_splits = []

        # Reverse map to convert IDs back to strings later
        id_to_action = {v: k for k, v in self.action_tokens.items()}

        for i in range(len(hands)):
            legal_acts = []
            if not can_check[i]: legal_acts.append('fold')
            if can_check[i]: legal_acts.append('check')
            if can_call[i] and not call_is_allin[i]: legal_acts.append('call')
            if can_raise[i]: legal_acts.append('raise')
            if max_bets[i] > 0 or call_is_allin[i]: legal_acts.append('allin')

            # Fallback failsafe
            if not legal_acts: legal_acts = ['fold']

            hand_act_splits.append(len(legal_acts))
            for act in legal_acts:
                flat_b_indices.append(i)
                flat_act_ids.append(self.action_tokens[act])

        total_branches = len(flat_b_indices)

        # Build [<unk>, <action>] inputs for all valid branches simultaneously
        suffix_ids = torch.empty((total_branches, 2), dtype=torch.long, device=self.device)
        suffix_ids[:, 0] = self.unk_token_id
        suffix_ids[:, 1] = torch.tensor(flat_act_ids, device=self.device)

        # ---------------------------------------------------------
        # 3. EXPAND PAST_KEY_VALUES AND RUN SHALLOW FORWARD PASS
        # ---------------------------------------------------------
        b_idx_tensor = torch.tensor(flat_b_indices, device=self.device)

        expanded_cache_data = []
        for layer in pkv.layers:
            # 1. Slice the native keys and values directly
            expanded_k = layer.keys.index_select(0, b_idx_tensor)
            expanded_v = layer.values.index_select(0, b_idx_tensor)

            # 2. Preserve sliding window tensors if the model uses them
            sliding_tensor = getattr(layer, "_sliding_window_tensor", None)

            if sliding_tensor is not None:
                expanded_cache_data.append((expanded_k, expanded_v, sliding_tensor))
            else:
                expanded_cache_data.append((expanded_k, expanded_v))

        # 3. Natively initialize a new cache using the ddp_cache_data constructor
        expanded_pkv = DynamicCache(expanded_cache_data)

        # Expand attention mask and add two 1s for the suffix tokens
        base_mask_expanded = attention_mask.index_select(0, b_idx_tensor)
        suffix_mask = torch.ones((total_branches, 2), device=self.device, dtype=attention_mask.dtype)
        branch_mask = torch.cat([base_mask_expanded, suffix_mask], dim=1)

        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            branch_outputs = self.model(
                input_ids=suffix_ids,
                attention_mask=branch_mask,
                past_key_values=expanded_pkv
            )

        branch_logits = branch_outputs.logits.float()

        # ---------------------------------------------------------
        # 4. EXTRACT DISTANCES AND REGRET MATCHING
        # ---------------------------------------------------------
        # Logits at the <action> token predicting distance sizes
        bucket_logits = branch_logits[:, 1, :]

        start_id = self.min_size_token_id
        num_sizes = len(self.sizes_floats)
        size_logits = bucket_logits[:, start_id: start_id + num_sizes]

        pred_bucket_probs = F.softmax(size_logits, dim=-1)
        expected_distances = torch.sum(pred_bucket_probs * self.torch_sizes_float, dim=-1)

        # Split the flattened distances back into their original hand groups
        split_distances = torch.split(expected_distances, hand_act_splits)
        split_act_ids = torch.split(torch.tensor(flat_act_ids, device=self.device), hand_act_splits)

        final_actions = [''] * len(hands)
        final_sizes = [0] * len(hands)
        raise_indexes = []

        for i in range(len(hands)):
            dist = split_distances[i]
            act_ids = split_act_ids[i]

            # Distance to Regret Inversion: Regret = Mean(Dist) - Dist
            mean_dist = dist.mean()
            pred_regrets = mean_dist - dist
            pos_regrets = torch.clamp(pred_regrets, min=0.0)
            sum_pos = pos_regrets.sum()

            if sum_pos > 0:
                probs = pos_regrets / sum_pos
            else:
                probs = torch.ones_like(pos_regrets) / len(pos_regrets)

            chosen_idx = torch.multinomial(probs, num_samples=1).item()
            chosen_act_id = act_ids[chosen_idx].item()
            chosen_act = id_to_action[chosen_act_id]

            final_actions[i] = chosen_act

            if chosen_act == 'allin':
                final_sizes[i] = max_bets[i]
            elif chosen_act == 'raise':
                raise_indexes.append(i)

        # Cleanup memory before raise sizing pass
        del inputs, input_ids, attention_mask, outputs, pkv
        del suffix_ids, expanded_pkv, branch_mask, branch_outputs, branch_logits

        # ---------------------------------------------------------
        # 5. EXISTING RAISE SIZING LOGIC
        # ---------------------------------------------------------
        if raise_indexes:
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

    def verify_and_log_state(self,text, b_idx, acts, scores, valid_probs, targets, loss, weight):
        """Helper to print out exactly what the model sees and targets for one sample."""
        print(f"\n--- STATE AUTOMATA VERIFICATION (Batch Index {b_idx}) ---")
        print(f"text: {text}")
        print(f"Legal Actions available: {acts}")
        print(f"Raw EV Scores: {scores}")
        print(f"Current Model Probs (Normalized over valid): {valid_probs.tolist()}")
        print(f"Assigned Targets: {targets.tolist()}")
        print(f"Node EV Spread Weight: {weight:.4f}")
        print(f"Resulting Unreduced CE Loss: {loss.item():.4f}")
        print("---------------------------------------------------------")


if __name__ == '__main__':
    sim = Simulator()

    # Standard queue, thread-safe
    experience_queue = queue.Queue(maxsize=sim.batch_size * 2)
    global_train_count = ThreadSafeCounter()

    num_generators = 4
    threads = []

    # Spawn standard threads
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
