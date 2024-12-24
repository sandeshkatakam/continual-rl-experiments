from abc import ABC, abstractmethod
import torch
import numpy as np

class BaseAgent(ABC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def select_action(self, state):
        """Select an action given the current state."""
        pass
    
    @abstractmethod
    def update(self, experience_batch):
        """Update the agent's parameters."""
        pass
    
    @abstractmethod
    def save(self, path):
        """Save the agent's parameters."""
        pass
    
    @abstractmethod
    def load(self, path):
        """Load the agent's parameters."""
        pass