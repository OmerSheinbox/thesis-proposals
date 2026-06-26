import numpy as np
from scipy.stats import entropy
from typing import List, Dict

class ContextGatheringDecisionProcess:
    """
    Implements CGDP: The agent's sole intrinsic reward is the reduction of Shannon entropy 
    over the hidden task specification space (Expected Information Gain).
    """
    def __init__(self, task_space_size: int):
        # Initialize a uniform belief over the task space
        self.belief = np.ones(task_space_size) / task_space_size
        
    def current_entropy(self) -> float:
        """Calculates the Shannon entropy of the current belief state."""
        # Using base 2 for bits of information
        return entropy(self.belief, base=2)

    def expected_information_gain(self, action: str, possible_observations: List[Dict]) -> float:
        """
        Simulates the expected reduction in entropy if a specific question (action) is asked.
        """
        # Placeholder: simulating that this action halves the entropy
        current_h = self.current_entropy()
        expected_future_h = current_h * 0.5  
        return current_h - expected_future_h

    def select_optimal_question(self, available_questions: List[str]) -> str:
        """
        Selects the question that maximizes Expected Information Gain.
        """
        best_q = None
        max_eig = -1.0
        
        for q in available_questions:
            # Fake observation set for placeholder
            eig = self.expected_information_gain(q, [{"obs": 1}, {"obs": 2}])
            if eig > max_eig:
                max_eig = eig
                best_q = q
                
        return best_q

if __name__ == "__main__":
    # Task space with 1024 possible specifications
    cgdp = ContextGatheringDecisionProcess(task_space_size=1024)
    print(f"Initial Entropy: {cgdp.current_entropy():.2f} bits")
    
    questions = ["What operating system are you on?", "Is the light blinking red or blue?"]
    best_question = cgdp.select_optimal_question(questions)
    
    print(f"Selected Question to Maximize EIG: '{best_question}'")
