import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import copy
from random import random, randint
import schedulefree
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from sim_hand import Hand
import torch
import numpy as np
from sim_encoder import Encoder
import concurrent.futures
import queue
import threading
import gc


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

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained('./opt-it-2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # Token definitions
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

        # Sizes setup
        self.sizes = list(range(1, 5))
        self.sizes = np.int16(self.sizes)
        self.torch_sizes = torch.tensor(self.sizes).to(self.device)
        self.torch_sizes_float = self.torch_sizes.float()
        self.sizes_floats = np.array(self.torch_sizes_float.tolist(), dtype=np.float32)

        self.encoder = Encoder()
        self.loss = torch.nn.KLDivLoss(reduction="sum", log_target=True).to(self.device)
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=1e-5)
        self.optimizer.train()

        self.n_sims = 8
        self.batch_size = 8
        self.num_generators = 5
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        self.local_storage = threading.local()

        self.batch_queue = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.global_updates = 0

    def get_local_encoders(self, required_count):
        if not hasattr(self.local_storage, 'encoders'):
            self.local_storage.encoders = [Encoder() for _ in range(32)]
        while len(self.local_storage.encoders) < required_count:
            self.local_storage.encoders.append(Encoder())
        return self.local_storage.encoders

    def data_generator_worker(self, worker_id):

        itr_street_cutoffs = [3, 2, 1, 0]
        hand_street_cutoffs = [5, 4, 3, 0]

        buffer_text = []
        buffer_evs = []

        while not self.stop_event.is_set():
            updates = self.global_updates
            itr_street_index = 0
            if updates > 0:
                idx = (updates // 2000)
                itr_street_index = min(idx, len(itr_street_cutoffs) - 1)

            while len(buffer_text) < self.batch_size and not self.stop_event.is_set():
                hands, hero_ev_data, hero_indices = self.batch_generate_hands(
                    3*self.batch_size,
                    itr_street_cutoffs[itr_street_index]
                )

                for i, hand in enumerate(hands):
                    if len(hero_ev_data[i]) > 0:
                        hero_idx = hero_indices[i]
                        uh = hand.get_u_hand(hero_idx)
                        if len(uh[1]) >= hand_street_cutoffs[itr_street_index]:
                            buffer_text.append(self.encoder.encode(json.dumps(uh), True))
                            buffer_evs.append(hero_ev_data[i])

                # Cleanup
                del hands, hero_ev_data, hero_indices

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

        print(f"Starting Training with {self.num_generators} Generators (Memory Optimized)...")

        try:
            while self.global_updates < 8000:
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

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
                    target_probs = torch.softmax(ref_outputs.logits, dim=2)

                ev_indexes = torch.zeros(len(batch_text), dtype=torch.long).to(self.device)
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
                            idx = idx.item()
                            if ev_indexes[idx] < len(batch_hero_evs[idx]):
                                evs = batch_hero_evs[idx][ev_indexes[idx].item()]
                                ev_indexes[idx] += 1

                                action_strs = list(evs.keys())
                                vals = np.array(list(evs.values())) / .5
                                max_val = np.max(vals)
                                exp_vals = np.exp(vals - max_val)
                                dist_probs = exp_vals / np.sum(exp_vals)

                                target_probs[idx, t - 1] = 0
                                for j, act_str in enumerate(action_strs):
                                    if act_str in action_map:
                                        target_probs[idx, t - 1, action_map[act_str]] = max(float(dist_probs[j]), 1e-9)
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

                # Manual Deallocation
                del logits, target_probs, shift_logits, shift_target_probs, loss, final_loss
                del outputs, ref_outputs, input_ids, attention_mask, hero_ids
              
                self.global_updates += 1
                if self.global_updates % 10 == 0:
                    print(f"Update {self.global_updates} | Loss: {np.mean(losses):.6f} | Q: {self.batch_queue.qsize()}")
                    losses = []

                    if self.global_updates % 100 == 0:
                        # Clear cache occasionally
                        torch.cuda.empty_cache()
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

        for i, hand in enumerate(hand_list):
            action_space = hand.get_action_space()
            raise_size = size_list[i]
            player = hand.state.turn_index
            res = {}
            if 'fold' in action_space: res['fold'] = -hand.investment()

          registry.append({'holder': res, 'is_calc': False})

            for root_action in action_space.keys():
                if root_action == 'fold': continue

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
                    res[root_action] = p[player]
                else:
                    start_idx = len(all_sims)
                    for _ in range(self.n_sims): all_sims.append(copy.deepcopy(temp_hand))
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
                current_holder[act] = np.mean(vals)
        if current_holder is not None: final_output.append(current_holder)

        return final_output

    @torch.inference_mode()
    def select_action_batch(self, hands):
        if not hands: return [], []
        local_encoders = self.get_local_encoders(len(hands))
        results = list(self.executor.map(self.process_hand_cpu, zip(hands, local_encoders[:len(hands)])))
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, can_check, can_raise = zip(*results)
      
        inputs = self.tokenizer(list(encoded_strs), padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        actor_tokens = input_ids[:, 1]
        input_ids = torch.cat([input_ids, actor_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((len(hands), 1), device=self.device)], dim=1)

        logits = self.model(input_ids, attention_mask=attention_mask).logits[:, -1, :]

        action_logits = torch.full_like(logits, float('-inf'))
        mask_check = torch.tensor(can_check, device=self.device)
        mask_raise = torch.tensor(can_raise, device=self.device)

        action_logits[mask_check, self.check_token] = logits[mask_check, self.check_token]
        action_logits[~mask_check, self.fold_token] = logits[~mask_check, self.fold_token]
        action_logits[~mask_check, self.call_token] = logits[~mask_check, self.call_token]
        action_logits[mask_raise, self.raise_token] = logits[mask_raise, self.raise_token]
        action_logits[mask_raise, self.allin_token] = logits[mask_raise, self.allin_token]

        probs = torch.softmax(action_logits, dim=1)
        action_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        del logits, action_logits, probs, input_ids, attention_mask, inputs
        del mask_check, mask_raise, actor_tokens

        action_tokens_cpu = action_tokens.tolist()
        del action_tokens

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
                r_inputs = self.tokenizer(r_encoded, padding=True, return_tensors="pt")
                r_ids = r_inputs.input_ids.to(self.device, non_blocking=True)
                r_mask = r_inputs.attention_mask.to(self.device, non_blocking=True)

                r_actor = r_ids[:, 1]
                r_raise_tok = torch.full((len(raise_idxs), 1), self.raise_token_id, device=self.device)

                r_final_ids = torch.cat([r_ids, r_actor.unsqueeze(1), r_raise_tok], dim=1)
                r_ext_mask = torch.ones((len(raise_idxs), 2), device=self.device)
                r_final_mask = torch.cat([r_mask, r_ext_mask], dim=1)

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

                del r_inputs, r_ids, r_mask, r_actor, r_raise_tok
                del r_final_ids, r_ext_mask, r_final_mask, r_logits
                del size_logits, r_min_tokens, offsets, size_indices, mask, size_probs
                del r_indices, r_chosen_pct, r_pots, r_bets

        return final_actions, final_sizes

    @torch.inference_mode()
    def select_raise_batch(self, hands):
        if not hands: return []
        local_encoders = self.get_local_encoders(len(hands))
        results = list(self.executor.map(self.process_hand_cpu, zip(hands, local_encoders[:len(hands)])))
        encoded_strs, min_bet_tokens, max_bets, pot_sizes, _, _ = zip(*results)

        inputs = self.tokenizer(list(encoded_strs), padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)

        actor_tokens = input_ids[:, 1]
        raise_col = torch.full((len(hands), 1), self.raise_token_id, device=self.device, dtype=torch.long)

        final_input_ids = torch.cat([input_ids, actor_tokens.unsqueeze(1), raise_col], dim=1)
        extension_mask = torch.ones((len(hands), 2), device=self.device, dtype=torch.long)
        final_attention_mask = torch.cat([attention_mask, extension_mask], dim=1)

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

        del inputs, input_ids, attention_mask, actor_tokens, raise_col
        del final_input_ids, extension_mask, final_attention_mask, logits
        del size_logits, batch_min_tokens, offsets, size_indices, mask, probs
        del relative_indices, chosen_percents, batch_pots, bets

        for i, amt in enumerate(bets_list):
            cap = max_bets[i]
            if cap > 0: amt = min(amt, cap)
            final_bets.append(int(amt))

        return final_bets

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

        u_hand = hand.get_u_hand()
        encoded_str = encoder.encode(json.dumps(u_hand))
        can_check = 'check' in action_space
        can_raise = 'raise' in action_space

        return encoded_str, min_bet_token, max_bet, pot_size, can_check, can_raise


if __name__ == '__main__':
    sim = Simulator()
    sim.rl()
