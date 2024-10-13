import torch
import math
from collections import deque
from tqdm import tqdm
from .utils import get_unique_states, state2hash
from .model import batch_process

class Searcher:
    def __init__(self, n, distance_metric='model', model=None, device=None, verbose=0):
        self.model = model.to(device) if model is not None else None
        self.n = n
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.distance_metric = distance_metric

        # Compute N and initialize V0
        N = 3 * n**2 - 3 * n + 1
        self.N = N
        self.V0 = torch.arange(N, dtype=torch.uint8).unsqueeze(0).to(self.device)

        # Load generators
        with open(f"generators/{n}.txt") as f:
            generators_str = f.readline()
        self.generators = torch.tensor(eval(generators_str), dtype=torch.int64).to(self.device)

        # Load manhattan_matrix
        with open(f"maps/{n}_manhattan.txt") as f:
            manhattan_str = f.readline()
        self.manhattan_matrix = torch.tensor(eval(manhattan_str), dtype=torch.int64).to(self.device)

        self.state_size = self.V0.size(1)
        self.hash_vec = torch.randint(0, 10**int(math.log10(2**62 / (self.N * self.N))), (self.N,), device=self.device)

    def hamming_dist(self, states):
        """Calculate the Hamming distance between the current states and the goal state."""
        return self.N - (states == self.V0).sum(dim=1)

    def manhattan_dist(self, states):
        """Calculate the Manhattan distance for the given states."""
        return torch.gather(self.manhattan_matrix, 1, torch.argsort(states, dim=1).T).sum(dim=0)

#     def get_neighbors(self, J, states, zeropos, batch_size=2**14):
#         """Generate neighboring states for each state in the batch."""
#         total_size = states.size(0)
#         state_size = states.size(1)

#         neighbor_states = torch.empty(total_size * 6, state_size, device=self.device, dtype=states.dtype)
#         zeropos_neighbors = torch.empty(total_size * 6, device=self.device, dtype=zeropos.dtype)

#         for i in range(0, total_size, batch_size):
#             batch_states = states[i:i + batch_size]
#             batch_zeropos = zeropos[i:i + batch_size]
#             batch_size_actual = batch_states.size(0)

#             # Create all_permutations for the current batch
#             all_permutations = 2 * torch.arange(6, device=self.device).unsqueeze(0).repeat(batch_size_actual, 1)
#             if J % 2 == 0:
#                 all_permutations += 1

#             from_positions, to_positions = self.generators[batch_zeropos.unsqueeze(1), all_permutations].unbind(2)
#             expanded_states = batch_states.unsqueeze(1).expand(-1, 6, -1).clone()
#             moved_values = torch.gather(expanded_states, 2, from_positions)
#             expanded_states.scatter_(2, to_positions, moved_values)
#             zeropos_neighbors_batch = torch.sum((moved_values == 0) * to_positions, dim=2).long()
#             start_idx = i * 6
#             end_idx = start_idx + batch_size_actual * 6
#             neighbor_states[start_idx:end_idx] = expanded_states.view(-1, state_size)
#             zeropos_neighbors[start_idx:end_idx] = zeropos_neighbors_batch.view(-1)
#         return neighbor_states, zeropos_neighbors
    def get_neighbors(self, J, states, zeropos, batch_size=2**14):
        total_size = states.size(0)
        state_size = states.size(1)

        neighbor_states = torch.empty(total_size, 6, state_size, device=states.device, dtype=states.dtype)
        zeropos_neighbors = torch.empty(total_size, 6, device=states.device, dtype=zeropos.dtype)

        for i in range(0, total_size, batch_size):
            batch_states = states[i:i + batch_size]
            batch_zeropos = zeropos[i:i + batch_size]
            batch_size_actual = batch_states.size(0)

            # Создаем all_permutations для текущего батча
            all_permutations = 2 * torch.arange(6, device=states.device).repeat(batch_size_actual, 1)
            if J % 2 == 0: all_permutations += 1

            from_positions, to_positions = self.generators[batch_zeropos.unsqueeze(1), all_permutations].unbind(2)
            expanded_states = batch_states.unsqueeze(1).expand(-1, 6, -1).clone()        
            moved_values = torch.gather(expanded_states, 2, from_positions)
            expanded_states.scatter_(2, to_positions, moved_values)
            zeropos_neighbors_batch = torch.sum((moved_values == 0) * to_positions, dim=2).long()
            neighbor_states[i:i + batch_size] = expanded_states
            zeropos_neighbors[i:i + batch_size] = zeropos_neighbors_batch

        return neighbor_states, zeropos_neighbors

    def do_greedy_step(self, J, states, zeropos, states_bad_hashed, B=1000):
        """Perform a greedy step to find the best neighbors."""
        idx0 = torch.arange(states.size(0), device=self.device).repeat_interleave(6)
        moves = 2 * torch.arange(6, device=self.device).repeat(states.size(0))
        if J % 2 == 0:
            moves += 1

        neighbors, zeropos = self.get_neighbors(J, states, zeropos)
        neighbors = neighbors.flatten(end_dim=1)
        zeropos = zeropos.flatten(end_dim=1)

        # Remove duplicate and bad states
        neighbors, idx1, states_bad_hashed = get_unique_states(neighbors, states_bad_hashed, self.hash_vec)

        # Predict values for the neighboring states
        if self.distance_metric == 'hamming':
            value = self.hamming_dist(neighbors)
        elif self.distance_metric == 'manhattan':
            value = self.manhattan_dist(neighbors)
        elif self.distance_metric == 'model':
            value = self.pred_d(neighbors)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        idx2 = torch.argsort(value)[:B]

        return neighbors[idx2], zeropos[idx1][idx2], states_bad_hashed, value[idx2], moves[idx1[idx2]], idx0[idx1[idx2]]

    def check_stagnation(self, states_log):
        """Check if the process is in a stagnation state."""
        if len(states_log) < 4:
            return False
        recent_states = torch.cat(list(states_log)[-2:])
        previous_states = torch.cat(list(states_log)[:-2])
        return torch.isin(recent_states, previous_states).all().item()

    def get_solution(self, state, parity=0, B=2**12, num_steps=200, num_attempts=10):
        """Main solution-finding loop that attempts to solve the puzzle."""
        states_bad_hashed = torch.tensor([], dtype=torch.int64, device=self.device)
        for attempt in range(num_attempts):
            states = state.unsqueeze(0).clone().to(self.device)
            zeropos = torch.zeros(states.size(0), dtype=torch.long, device=self.device)  # Assuming zero starts at position 0
            tree_move = torch.zeros((num_steps, B), dtype=torch.int64, device=self.device)
            tree_idx = torch.zeros((num_steps, B), dtype=torch.int64, device=self.device)
            states_hash_log = deque(maxlen=4)

            if self.verbose:
                pbar = tqdm(range(num_steps), desc=f"Attempt {attempt+1}/{num_attempts}")
            else:
                pbar = range(num_steps)
            for j in pbar:
                states, zeropos, states_bad_hashed, value, moves, idx = self.do_greedy_step(
                    j+parity, 
                    states, zeropos, states_bad_hashed, B=B
                )
                if self.verbose:
                    pbar.set_description(f"Attempt {attempt+1}/{num_attempts}, y_min={value.min().item():.1f}, y_mean={value.float().mean().item():.1f}")
#                 states_hash_log.append(state2hash(states, self.hash_vec))
                leaves_num = states.size(0)
                tree_move[j, :leaves_num] = moves
                tree_idx[j, :leaves_num] = idx

                if (states == self.V0).all(dim=1).any():
                    break
#                 elif (j > 3 and self.check_stagnation(states_hash_log)):
#                     states_bad_hashed = torch.cat((states_bad_hashed, torch.cat(list(states_hash_log))))
#                     states_bad_hashed = torch.unique(states_bad_hashed)
#                     break

            if (states == self.V0).all(dim=1).any():
                break
        else:
            # If solution is not found after all attempts
            return None, attempt + 1

        # Reverse the tree to reconstruct the path
        tree_idx, tree_move = tree_idx[:j+1].flip((0,)), tree_move[:j+1].flip((0,))
        V0_pos = torch.nonzero((states == self.V0).all(dim=1), as_tuple=True)[0].item()

        # Construct the path
        path = [tree_idx[0, V0_pos].item()]
        for k in range(1, j+1):
            path.append(tree_idx[k, path[-1]].item())

        moves_seq = torch.tensor([tree_move[k, path[k-1]] if k > 0 else tree_move[k, V0_pos] for k in range(j+1)], dtype=torch.int64)
        return moves_seq.flip((0,)), attempt + 1

    def pred_d(self, states):
        """Predict values for states using the model."""
        if self.model is None:
            raise ValueError("Model is not provided for 'model' distance metric.")
        return batch_process(self.model, states, self.device, batch_size=2**14)

    def moves_seq_to_string(self, moves_seq):
        """Convert moves sequence to string representation using move labels."""
        move_labels = ["A", "1", "B", "2", "C", "3", "D", "4", "E", "5", "F", "6"]
        return ''.join([move_labels[j] for j in moves_seq.tolist()])
