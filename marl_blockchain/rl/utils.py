import jax
import jax.numpy as jnp
import optax
from flax import struct

import collections
import jax
import jax.numpy as jnp
import jraph


class ReplayBuffer:
    def __init__(self, capacity, graph_template):
        """
        Initializes the replay buffer for graph-based RL.

        Args:
            capacity (int): The maximum number of experiences the buffer can store.
            graph_template (jraph.GraphsTuple): A template for the graphs, defining the structure.
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.graph_template = graph_template

    def add(self, graph, action, reward, next_graph, done):
        """
        Adds a new experience to the buffer.

        Args:
            graph (jraph.GraphsTuple): The graph structure representing the current state.
            action (int or array): The action taken at the current state.
            reward (float): The reward received after taking the action.
            next_graph (jraph.GraphsTuple): The graph structure representing the next state.
            done (bool): Whether the episode has finished after this step.
        """
        experience = (graph, action, reward, next_graph, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A batch of experiences (graph, action, reward, next_graph, done).
        """
        indices = jax.random.randint(
            key=jax.random.PRNGKey(0),
            shape=(batch_size,),
            minval=0,
            maxval=len(self.buffer)
        )

        batch = [self.buffer[i] for i in indices]

        # Separate batch elements for easier processing
        graphs, actions, rewards, next_graphs, dones = zip(*batch)

        return (self._batch_graphs(graphs),
                jnp.array(actions),
                jnp.array(rewards),
                self._batch_graphs(next_graphs),
                jnp.array(dones))

    def _batch_graphs(self, graphs):
        """
        Batches multiple jraph.GraphsTuples into a single graph.

        Args:
            graphs (list of jraph.GraphsTuple): The list of graphs to batch.

        Returns:
            jraph.GraphsTuple: A batched graph structure.
        """
        return jraph.batch(graphs)

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)


class CentralizedReplayBuffer:
    """
    This class defines a Replay Buffer common to several experiments being run in parallel for optimized training.
    When an process samples from the buffer, it samples from the buffer of all process, with a majority of the samples from
    his own buffer.
    """

    def __init__(self, capacity, graph_template, num_processes):
        """
        Initializes the centralized replay buffer for graph-based RL.

        Args:
            capacity (int): The maximum number of experiences the buffer can store.
            graph_template (jraph.GraphsTuple): A template for the graphs, defining the structure.
            num_processes (int): The number of processes running in parallel.
        """
        self.capacity = capacity
        self.graph_template = graph_template
        self.num_processes = num_processes

        self.buffers = [ReplayBuffer(capacity // num_processes, graph_template) for _ in range(num_processes)]

    def add(self, process_id, graph, action, reward, next_graph, done):
        """
        Adds a new experience to the buffer of a specific process.

        Args:
            process_id (int): The ID of the process to add the experience to.
            graph (jraph.GraphsTuple): The graph structure representing the current state.
            action (int or array): The action taken at the current state.
            reward (float): The reward received after taking the action.
            next_graph (jraph.GraphsTuple): The graph structure representing the next state.
            done (bool): Whether the episode has finished after this step.
        """

        self.buffers[process_id].add(graph, action, reward, next_graph, done)

    def sample(self, batch_size, process_id):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.
            process_id (int): The ID of the process to sample the majority of experiences from.

        Returns:
            A batch of experiences (graph, action, reward, next_graph, done).
        """

        sub_batch_size = batch_size // 2

        # Sample batch from the current process
        batch = self.buffers[process_id].sample(sub_batch_size)

        # Sample and concatenate batches from other processes
        other_batches = [
            self.buffers[i].sample(sub_batch_size)
            for i in range(self.num_processes) if i != process_id
        ]

        # Concatenate batches using list comprehension
        batch = tuple(
            jnp.concatenate([batch[i]] + [other_batch[i] for other_batch in other_batches])
            for i in range(len(batch))
        )

        return batch

    def __len__(self):
        """
        Returns the current size of the buffer.
        :return:
        """
        return sum(len(buffer) for buffer in self.buffers)
