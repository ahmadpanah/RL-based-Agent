import numpy as np
import pandas as pd
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import random
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionConfig:
    cpu: float
    memory: int
    timeout: int
    max_instances: int
    min_instances: int = 1
    cold_start_delay: float = 2.0  # seconds
    warm_start_delay: float = 0.5  # seconds

class WorkloadGenerator:
    def __init__(self, mean_requests: int = 20, variance: int = 10):
        self.mean = mean_requests
        self.variance = variance
        self.current_time = 0
        
    def generate_workload(self, time_window: float) -> List[float]:
        """Generate Poisson-distributed workload for given time window"""
        num_requests = int(np.random.normal(self.mean, self.variance))
        return np.sort(np.random.uniform(0, time_window, num_requests))

class Request:
    def __init__(self, arrival_time: float, execution_time: float):
        self.arrival_time = arrival_time
        self.execution_time = execution_time
        self.start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        self.instance_id: Optional[int] = None
        self.is_cold_start: bool = False

class FunctionInstance:
    def __init__(self, instance_id: int):
        self.instance_id = instance_id
        self.state = "idle"
        self.current_request: Optional[Request] = None
        self.last_used_time: float = 0
        self.cpu_utilization = 0.0
        self.total_requests_handled = 0
        self.cold_starts = 0

class MetricsCollector:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.cpu_utilization = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.failure_rates = deque(maxlen=window_size)
        self.cold_starts = deque(maxlen=window_size)
        
    def add_metrics(self, cpu: float, response_time: float, 
                   failures: float, cold_starts: int):
        self.cpu_utilization.append(cpu)
        self.response_times.append(response_time)
        self.failure_rates.append(failures)
        self.cold_starts.append(cold_starts)
        
    def get_average_metrics(self) -> Dict[str, float]:
        return {
            'avg_cpu': np.mean(self.cpu_utilization),
            'avg_response_time': np.mean(self.response_times),
            'avg_failure_rate': np.mean(self.failure_rates),
            'avg_cold_starts': np.mean(self.cold_starts)
        }

class ServerlessEnvironment:
    def __init__(self, 
                 function_config: FunctionConfig,
                 metrics_window: int = 100):
        self.config = function_config
        self.instances: Dict[int, FunctionInstance] = {}
        self.request_queue = deque()
        self.metrics = MetricsCollector(metrics_window)
        self.current_time = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.cold_starts = 0
        self.workload_generator = WorkloadGenerator()
        
        # Initialize with minimum instances
        for i in range(self.config.min_instances):
            self.instances[i] = FunctionInstance(i)
    
    def process_workload(self, time_window: float) -> Dict[str, float]:
        """Process incoming workload for given time window"""
        requests = self.workload_generator.generate_workload(time_window)
        metrics = []
        
        for req_time in requests:
            self.current_time = req_time
            request = Request(req_time, random.uniform(0.5, 2.0))
            
            # Try to assign request to available instance
            instance = self._get_available_instance()
            if instance:
                self._process_request(request, instance)
            else:
                self.failed_requests += 1
            
            self.total_requests += 1
            
            # Collect metrics
            metrics.append(self._collect_current_metrics())
        
        return pd.DataFrame(metrics).mean().to_dict()
    
    def _get_available_instance(self) -> Optional[FunctionInstance]:
        """Get available instance or create new one if possible"""
        # First try to find idle instance
        for instance in self.instances.values():
            if instance.state == "idle":
                return instance
        
        # If no idle instance and can create new one
        if len(self.instances) < self.config.max_instances:
            new_id = len(self.instances)
            self.instances[new_id] = FunctionInstance(new_id)
            self.cold_starts += 1
            return self.instances[new_id]
        
        return None
    
    def _process_request(self, request: Request, instance: FunctionInstance):
        """Process request on given instance"""
        instance.state = "busy"
        instance.current_request = request
        instance.total_requests_handled += 1
        
        # Add cold start delay if instance hasn't been used recently
        if self.current_time - instance.last_used_time > 300:  # 5 minutes
            request.is_cold_start = True
            request.start_time = self.current_time + self.config.cold_start_delay
        else:
            request.start_time = self.current_time + self.config.warm_start_delay
            
        request.finish_time = request.start_time + request.execution_time
        instance.last_used_time = request.finish_time
        instance.cpu_utilization = random.uniform(0.4, 0.9)
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        active_instances = sum(1 for i in self.instances.values() if i.state == "busy")
        avg_cpu = np.mean([i.cpu_utilization for i in self.instances.values()])
        failure_rate = self.failed_requests / max(1, self.total_requests)
        
        return {
            'instances': len(self.instances),
            'active_instances': active_instances,
            'cpu_utilization': avg_cpu,
            'failure_rate': failure_rate,
            'cold_starts': self.cold_starts
        }

class RLAgent(ABC):
    @abstractmethod
    def select_action(self, state) -> int:
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state):
        pass

class QLearningAgent(RLAgent):
    def __init__(self, 
                 state_dims: int,
                 action_space: List[int],
                 learning_rate: float = 0.9,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.0025,
                 min_epsilon: float = 0.01):
        self.state_dims = state_dims
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table with optimistic initial values
        self.q_table = defaultdict(lambda: {a: 1.0 for a in action_space})
        
    def select_action(self, state) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation"""
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state][action] = new_value
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, 
                         self.epsilon * (1 - self.epsilon_decay))

def train_agent(epochs: int = 500, 
                iterations_per_epoch: int = 5,
                time_window: float = 120) -> Tuple[QLearningAgent, List[Dict]]:
    """Train the Q-learning agent"""
    
    function_config = FunctionConfig(
        cpu=1,
        memory=128,
        timeout=60,
        max_instances=7
    )
    
    env = ServerlessEnvironment(function_config)
    action_space = [-1, 0, 1]  # Decrease, no change, increase instances
    agent = QLearningAgent(state_dims=3, action_space=action_space)
    
    training_history = []
    
    for epoch in range(epochs):
        logger.info(f"Training epoch {epoch + 1}/{epochs}")
        epoch_metrics = []
        
        for iteration in range(iterations_per_epoch):
            # Get current state
            metrics = env._collect_current_metrics()
            current_state = (
                metrics['instances'],
                int(metrics['cpu_utilization'] * 100),
                int(metrics['failure_rate'] * 100)
            )
            
            # Select and perform action
            action = agent.select_action(current_state)
            
            # Process workload and get new state
            new_metrics = env.process_workload(time_window)
            next_state = (
                int(new_metrics['instances']),
                int(new_metrics['cpu_utilization'] * 100),
                int(new_metrics['failure_rate'] * 100)
            )
            
            # Calculate reward
            reward = (0.75 - new_metrics['cpu_utilization']) + \
                    (0.20 - new_metrics['failure_rate'])
            
            # Update agent
            agent.update(current_state, action, reward, next_state)
            
            # Store metrics
            epoch_metrics.append({
                'epoch': epoch,
                'iteration': iteration,
                **new_metrics,
                'reward': reward
            })
        
        training_history.extend(epoch_metrics)
        
        # Log epoch summary
        epoch_df = pd.DataFrame(epoch_metrics)
        logger.info(f"Epoch {epoch + 1} summary:")
        logger.info(f"Average reward: {epoch_df['reward'].mean():.4f}")
        logger.info(f"Average CPU utilization: {epoch_df['cpu_utilization'].mean():.2%}")
        logger.info(f"Average failure rate: {epoch_df['failure_rate'].mean():.2%}")
        logger.info(f"Total cold starts: {epoch_df['cold_starts'].sum()}")
        
    return agent, training_history

if __name__ == "__main__":
    # Train agent
    logger.info("Starting training...")
    trained_agent, history = train_agent(epochs=500, iterations_per_epoch=5)
    
    # Convert history to DataFrame for analysis
    history_df = pd.DataFrame(history)
    
    # Print final results
    logger.info("\nTraining completed. Final results:")
    logger.info(f"Average reward: {history_df['reward'].mean():.4f}")
    logger.info(f"Average CPU utilization: {history_df['cpu_utilization'].mean():.2%}")
    logger.info(f"Average failure rate: {history_df['failure_rate'].mean():.2%}")
    logger.info(f"Total cold starts: {history_df['cold_starts'].sum()}")