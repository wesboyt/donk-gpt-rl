import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans

import os
import sys
import copy
import json
import random
import queue
import threading
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from sim_hand import Hand
from sim_encoder import Encoder

#This script compares two models by their divergence in strategy, useful for monitoring RL training runs to understand the posterior relative to the prior.


class SimulatorBatched:
    def __init__(self, config_path_a: str, weights_path_a: str, config_path_b: str, weights_path_b: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Loading Model A onto {self.device}...")
        config_a = AutoConfig.from_pretrained(config_path_a)
        self.model_a = AutoModelForCausalLM.from_config(config_a)
        self.model_a.load_state_dict(torch.load(weights_path_a, map_location=self.device, weights_only=True), strict=False)
        self.model_a.to(self.device).eval()
        self.model_a.share_memory()

        print(f"Loading Model B onto {self.device}...")
        config_b = AutoConfig.from_pretrained(config_path_b)
        self.model_b = AutoModelForCausalLM.from_config(config_b)
        self.model_b.load_state_dict(torch.load(weights_path_b, map_location=self.device, weights_only=True), strict=False)
        self.model_b.to(self.device).eval()
        self.model_b.share_memory()

        self.tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # Action logic
        self.action_names = ['fold', 'check', 'call', 'raise', 'allin']
        self.action_to_idx = {name: i for i, name in enumerate(self.action_names)}

        self.fold_token_id = self.tokenizer.encode("<fold>")[0]
        self.check_token_id = self.tokenizer.encode("<check>")[0]
        self.call_token_id = self.tokenizer.encode("<call>")[0]
        self.raise_token_id = self.tokenizer.encode("<raise>")[0]
        self.allin_token_id = self.tokenizer.encode("<allin>")[0]
        self.min_size_token_id = self.tokenizer.encode("<b1%>")[0]

        # Sizing setups
        self.sizes = np.array(list(range(1, 5)) + list(range(5, 101, 5)) + list(range(125, 501, 25)), dtype=np.float32)
        self.torch_sizes_float = torch.tensor(self.sizes).to(self.device).float()
        self.sizes_floats = self.torch_sizes_float.tolist()

        # Architecture queues
        self.queue_a = queue.Queue()
        self.queue_b = queue.Queue()
        self.thread_local = threading.local()

    @property
    def encoder(self):
        if not hasattr(self.thread_local, 'encoder'):
            self.thread_local.encoder = Encoder()
        return self.thread_local.encoder

    def inference_server_loop(self, model, inf_queue, name):
        """Runs continuously to batch requests and perform GPU forward passes."""
        print(f"GPU Inference Server Thread '{name}' started.")
        while True:
            try:
                req = inf_queue.get()
                reqs = [req]
                total_queries = len(req['queries'])
                target_batch_size = 128

                # Group queries for batching
                while total_queries < target_batch_size:
                    try:
                        new_req = inf_queue.get_nowait()
                        reqs.append(new_req)
                        total_queries += len(new_req['queries'])
                    except queue.Empty:
                        break

                all_queries = []
                slices = []
                current_pos = 0

                for r in reqs:
                    n = len(r['queries'])
                    all_queries.extend(r['queries'])
                    slices.append((current_pos, current_pos + n))
                    current_pos += n

                if not all_queries:
                    continue

                inputs = self.tokenizer(all_queries, padding=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.device, non_blocking=True)
                attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

                with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :].float()

                # Distribute results
                for idx, r in enumerate(reqs):
                    s, e = slices[idx]
                    r['result']['data'] = logits[s:e]
                    r['event'].set()

            except Exception as e:
                print(f"Inference Server {name} Error: {e}")
                traceback.print_exc()

    @torch.no_grad()
    def _compute_raise_sizes_from_logits(self, last_logits, min_bet_tokens, max_bets, pot_sizes):
        start_id = self.min_size_token_id
        num_sizes = len(self.sizes_floats)
        size_logits = last_logits[:, start_id: start_id + num_sizes]

        batch_min_tokens = torch.tensor(min_bet_tokens, device=self.device)
        offsets = (batch_min_tokens - start_id).unsqueeze(1)
        size_indices = torch.arange(num_sizes, device=self.device).unsqueeze(0)

        # Mask invalid sizes and sample
        size_logits = size_logits.masked_fill(~(size_indices >= offsets), float('-inf'))
        probs = torch.softmax(size_logits, dim=1)
        probs = torch.nan_to_num(probs, nan=1e-5)

        relative_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
        chosen_percents = self.torch_sizes_float[relative_indices]
        bets = (torch.tensor(pot_sizes, device=self.device) * (chosen_percents / 100.0)).long()

        return [
            int(min(amt, cap) if cap > 0 else amt)
            for amt, cap in zip(bets.tolist(), max_bets)
        ]

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

    @torch.no_grad()
    def select_action_batch(self, hands, inf_queue):
        if not hands: return [], []

        results = [self.process_hand_cpu(hand) for hand in hands]
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices = zip(*results)

        action_queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}>" for i in range(len(hands))]
        raise_queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}><raise>" for i in range(len(hands))]

        evt_action, evt_raise = threading.Event(), threading.Event()
        res_action, res_raise = {}, {}

        # Submit query to specified model's server
        inf_queue.put({'queries': action_queries, 'event': evt_action, 'result': res_action})
        inf_queue.put({'queries': raise_queries, 'event': evt_raise, 'result': res_raise})

        evt_action.wait()
        evt_raise.wait()

        action_logits = res_action['data']
        raise_logits = res_raise['data']

        pre_sampled_sizes = self._compute_raise_sizes_from_logits(raise_logits, min_bet_tokens, max_bets, pot_sizes)

        # Slice logits to only the 5 specific action tokens
        vocab_indices = [self.fold_token_id, self.check_token_id, self.call_token_id, self.raise_token_id, self.allin_token_id]
        hero_action_logits = action_logits[:, vocab_indices]

        batch_size = len(hands)
        logits_mask = torch.full((batch_size, 5), float('-inf'), device=self.device)

        for i in range(batch_size):
            if not can_check[i]: logits_mask[i, 0] = 0.0  # fold
            if can_check[i]: logits_mask[i, 1] = 0.0  # check
            if can_call[i] and not call_is_allin[i]: logits_mask[i, 2] = 0.0  # call

            can_raise_flag = can_raise[i]
            can_allin_flag = max_bets[i] > 0 or call_is_allin[i]
            raise_sz = pre_sampled_sizes[i]

            if max_bets[i] > 0 and raise_sz > 0 and max_bets[i] > 4 * raise_sz:
                can_allin_flag = False
            elif max_bets[i] > 0 and raise_sz > 0 and raise_sz >= (0.5 * max_bets[i]):
                can_raise_flag = False

            if can_raise_flag: logits_mask[i, 3] = 0.0  # raise
            if can_allin_flag: logits_mask[i, 4] = 0.0  # allin

            if (logits_mask[i] == float('-inf')).all():
                logits_mask[i, 0] = 0.0

        # Vectorized probability calculation & sampling
        masked_logits = hero_action_logits + logits_mask
        probs = torch.softmax(masked_logits, dim=-1)
        probs = torch.clamp(probs, min=1e-5) * (logits_mask == 0.0).float()
        probs = probs / probs.sum(dim=-1, keepdim=True)

        chosen_indices = torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()
        final_actions = [self.action_names[idx] for idx in chosen_indices]

        final_sizes = [
            max_bets[i] if act == 'allin' else (pre_sampled_sizes[i] if act == 'raise' else 0)
            for i, act in enumerate(final_actions)
        ]

        return final_actions, final_sizes

    def apply_action(self, hand, action, size):
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

    def run_batched_rollouts(self, hands, hero_model_id, hero_idx):
        active_indices = list(range(len(hands)))
        finished_payoffs = [0.0] * len(hands)

        while active_indices:
            hands_a = []
            hands_b = []

            for idx in active_indices:
                h = hands[idx]
                model_for_turn = hero_model_id if h.state.turn_index == hero_idx else ('B' if hero_model_id == 'A' else 'A')

                if model_for_turn == 'A':
                    hands_a.append(h)
                else:
                    hands_b.append(h)

            if hands_a:
                actions, sizes = self.select_action_batch(hands_a, self.queue_a)
                for h, act, sz in zip(hands_a, actions, sizes):
                    self.apply_action(h, act, sz)

            if hands_b:
                actions, sizes = self.select_action_batch(hands_b, self.queue_b)
                for h, act, sz in zip(hands_b, actions, sizes):
                    self.apply_action(h, act, sz)

            next_active = []
            for idx in active_indices:
                if hands[idx].done:
                    p = hands[idx].state.payoffs
                    for j in range(len(p)):
                        if p[j] > 0: p[j] -= min(p[j] * .05, 2 * hands[idx].big_blind)
                    finished_payoffs[idx] = p[hero_idx]
                else:
                    next_active.append(idx)

            active_indices = next_active

        return finished_payoffs

    def evaluate_hand(self, base_hand, n_sims=16):
        """Calculates Delta EV of Model A - Model B for a specific gamestate."""
        hero_idx = base_hand.state.turn_index

        hands_a_hero = [copy.deepcopy(base_hand) for _ in range(n_sims)]
        hands_b_hero = [copy.deepcopy(base_hand) for _ in range(n_sims)]

        payoffs_a = self.run_batched_rollouts(hands_a_hero, hero_model_id='A', hero_idx=hero_idx)
        payoffs_b = self.run_batched_rollouts(hands_b_hero, hero_model_id='B', hero_idx=hero_idx)

        ev_a = np.mean(payoffs_a) / base_hand.big_blind
        ev_b = np.mean(payoffs_b) / base_hand.big_blind

        return ev_a - ev_b

    @torch.no_grad()
    def get_action_distributions(self, hands, inf_queue, chunk_size=128):
        """Extracts the exact 5-dim action probability vector, chunked to prevent VRAM OOM."""
        if not hands: return np.array([])

        all_probs = []

        # Process the hands in chunks
        for chunk_start in range(0, len(hands), chunk_size):
            chunk_hands = hands[chunk_start : chunk_start + chunk_size]

            results = [self.process_hand_cpu(hand) for hand in chunk_hands]
            encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise, can_call, call_is_allin, actor_indices = zip(*results)

            action_queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}>" for i in range(len(chunk_hands))]
            raise_queries = [f"{encoded_strs[i]}<herop{actor_indices[i]}><raise>" for i in range(len(chunk_hands))]

            evt_action, evt_raise = threading.Event(), threading.Event()
            res_action, res_raise = {}, {}

            inf_queue.put({'queries': action_queries, 'event': evt_action, 'result': res_action})
            inf_queue.put({'queries': raise_queries, 'event': evt_raise, 'result': res_raise})

            evt_action.wait()
            evt_raise.wait()

            action_logits = res_action['data']
            raise_logits = res_raise['data']

            pre_sampled_sizes = self._compute_raise_sizes_from_logits(raise_logits, min_bet_tokens, max_bets, pot_sizes)

            vocab_indices = [self.fold_token_id, self.check_token_id, self.call_token_id, self.raise_token_id, self.allin_token_id]
            hero_action_logits = action_logits[:, vocab_indices]

            batch_size = len(chunk_hands)
            logits_mask = torch.full((batch_size, 5), float('-inf'), device=self.device)

            for i in range(batch_size):
                if not can_check[i]: logits_mask[i, 0] = 0.0
                if can_check[i]: logits_mask[i, 1] = 0.0
                if can_call[i] and not call_is_allin[i]: logits_mask[i, 2] = 0.0

                can_raise_flag = can_raise[i]
                can_allin_flag = max_bets[i] > 0 or call_is_allin[i]
                raise_sz = pre_sampled_sizes[i]

                if max_bets[i] > 0 and raise_sz > 0 and max_bets[i] > 4 * raise_sz:
                    can_allin_flag = False
                elif max_bets[i] > 0 and raise_sz > 0 and raise_sz >= (0.5 * max_bets[i]):
                    can_raise_flag = False

                if can_raise_flag: logits_mask[i, 3] = 0.0
                if can_allin_flag: logits_mask[i, 4] = 0.0

                if (logits_mask[i] == float('-inf')).all():
                    logits_mask[i, 0] = 0.0

            masked_logits = hero_action_logits + logits_mask
            probs = torch.softmax(masked_logits, dim=-1)
            probs = torch.clamp(probs, min=1e-5) * (logits_mask == 0.0).float()
            probs = probs / probs.sum(dim=-1, keepdim=True)

            all_probs.append(probs.cpu().numpy())

        # Stack all the chunks back together into a single numpy array
        return np.vstack(all_probs)


# --- NEW ANALYSIS PIPELINE ---

def generate_sbvbb_node_data(simulator, n_hands, n_sims_per_node):
    """Generates isolated SBvBB data, extracts probs, divergence, and EV."""
    data = []

    # Generate base hands
    hands = []
    for _ in range(n_hands):
        h = Hand()
        for _ in range(4):  # Fold around to SB
            h.fold()
        hands.append(h)

    print("Extracting Action Distributions from Model A...")
    probs_a = simulator.get_action_distributions(hands, simulator.queue_a)

    print("Extracting Action Distributions from Model B...")
    probs_b = simulator.get_action_distributions(hands, simulator.queue_b)

    print(f"Running Monte Carlo EV Rollouts ({n_sims_per_node} sims per node)...")

    # We evaluate EV delta using the existing simulator logic
    # Note: For massive scale, this should be chunked/batched.
    ev_deltas = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(simulator.evaluate_hand, h, n_sims_per_node) for h in hands]
        for i, future in enumerate(futures):
            ev_deltas.append(future.result())
            if (i + 1) % 50 == 0:
                print(f"Evaluated {i + 1}/{n_hands} nodes...")

    # Compile data
    for i in range(n_hands):
        p_a = probs_a[i]
        p_b = probs_b[i]

        # Calculate Jensen-Shannon Divergence
        js_div = distance.jensenshannon(p_a, p_b) ** 2

        # Calculate Strategy Variance (Diff vector)
        diff_vector = p_a - p_b

        sb_idx = hands[i].state.turn_index
        # Replace with this:
        hole_cards = hands[i].u_hand[0][sb_idx]

        data.append({
            'cards': str(hole_cards),
            'p_a_fold': p_a[0], 'p_a_check': p_a[1], 'p_a_call': p_a[2], 'p_a_raise': p_a[3], 'p_a_allin': p_a[4],
            'p_b_fold': p_b[0], 'p_b_check': p_b[1], 'p_b_call': p_b[2], 'p_b_raise': p_b[3], 'p_b_allin': p_b[4],
            'diff_fold': diff_vector[0], 'diff_check': diff_vector[1], 'diff_call': diff_vector[2],
            'diff_raise': diff_vector[3], 'diff_allin': diff_vector[4],
            'js_divergence': js_div,
            'delta_ev': ev_deltas[i]
        })

    return pd.DataFrame(data)


def analyze_strategy_clusters(df, n_clusters=5):
    """Clusters nodes based on strategy variance and extracts the barycenters."""

    # We cluster on the Difference Vector (P_A - P_B)
    diff_cols = ['diff_fold', 'diff_check', 'diff_call', 'diff_raise', 'diff_allin']
    X = df[diff_cols].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(X)

    print("\n" + "=" * 50)
    print("STRATEGY VARIANCE CLUSTERING REPORT")
    print("=" * 50)

    for cluster_idx in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_idx]
        barycenter = kmeans.cluster_centers_[cluster_idx]

        mean_div = cluster_data['js_divergence'].mean()
        mean_ev = cluster_data['delta_ev'].mean()
        node_count = len(cluster_data)

        print(f"\n### CLUSTER {cluster_idx} (Nodes: {node_count} | Mean JSD: {mean_div:.4f} | Mean ΔEV: {mean_ev:.4f} BB)")
        print("Barycenter Description (Model A preference vs Model B preference):")

        actions = ['Fold', 'Check', 'Call', 'Raise', 'All-in']
        for i, act in enumerate(actions):
            val = barycenter[i]
            if abs(val) > 0.05:  # Only show significant variance (>5%)
                direction = "A prefers" if val > 0 else "B prefers"
                print(f"  - {direction} {act} by {abs(val) * 100:.1f}%")

        # Show top 3 highest divergence hands in this cluster
        top_hands = cluster_data.sort_values('js_divergence', ascending=False).head(3)
        print(f"Example Hands (High Divergence): {', '.join(top_hands['cards'].tolist())}")


if __name__ == '__main__':
    sim = SimulatorBatched('config.json', 'GEN-17600000.pt', 'config.json', 'RL-75000.pt')

    threading.Thread(target=sim.inference_server_loop, args=(sim.model_a, sim.queue_a, 'A'), daemon=True).start()
    threading.Thread(target=sim.inference_server_loop, args=(sim.model_b, sim.queue_b, 'B'), daemon=True).start()

    # Parameters for analysis
    TOTAL_SBVBB_NODES = 10000  # Adjust based on memory/time
    SIMS_PER_NODE = 8  # Number of rollouts to calculate Delta EV

    print(f"Generating data for {TOTAL_SBVBB_NODES} SBvBB nodes...")
    df_analysis = generate_sbvbb_node_data(sim, TOTAL_SBVBB_NODES, SIMS_PER_NODE)

    # Save raw data just in case
    df_analysis.to_csv("sbvbb_divergence_data.csv", index=False)

    # Run the clustering
    analyze_strategy_clusters(df_analysis, n_clusters=30)
