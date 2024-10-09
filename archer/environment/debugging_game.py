import random
from typing import Optional, List, Tuple, Dict
import logging
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import concurrent.futures

# Set logging level to CRITICAL to suppress unnecessary logs
logging.getLogger().setLevel(logging.CRITICAL)

PROMPT_TEMPLATE = (
    'You are playing a game where at each round you are presented with several options. '
    'Choose the correct option to navigate towards the goal. '
    'For example, if given the options "Options: Go to Forest, Go to Lake", you would respond with "Go to Forest" or "Go to Lake". '
    'Game history:\n{history}\n'
    'Options: {options}\n'
    'Your choice:'
)

INITIAL_STR = "Game History:\n"

# Define a simple set of game scenarios with fixed options and terminal types
DEFAULT_GAME_SCENARIOS = {
    "Start": {
        "description": "You are at the start of the game.",
        "options": ["Go to Forest", "Go to Lake"],
        "responses": {
            "Go to Forest": "You have entered the Forest.",
            "Go to Lake": "You have arrived at the Lake."
        },
        "next": {
            "Go to Forest": {
                "options": ["Climb Tree", "Follow Path"],
                "responses": {
                    "Climb Tree": "You climbed a tree and found a hidden key.",
                    "Follow Path": "You followed the path and encountered a wild animal."
                },
                "next": {
                    "Climb Tree": {
                        "options": ["Use Key", "Return"],
                        "responses": {
                            "Use Key": "You used the key to unlock a treasure chest and won the game!",
                            "Return": "You returned to the forest entrance."
                        },
                        "next": {
                            "Use Key": {"terminal_type": "good"},  # Terminal state: Good
                            "Return": {"terminal_type": "neutral"}  # Terminal state: Neutral
                        }
                    },
                    "Follow Path": {
                        "options": ["Fight Animal", "Run Away"],
                        "responses": {
                            "Fight Animal": "You fought bravely but lost the battle.",
                            "Run Away": "You successfully escaped back to the forest entrance."
                        },
                        "next": {
                            "Fight Animal": {"terminal_type": "bad"},  # Terminal state: Bad
                            "Run Away": {"terminal_type": "neutral"}  # Terminal state: Neutral
                        }
                    }
                }
            },
            "Go to Lake": {
                "options": ["Swim Across", "Build Raft"],
                "responses": {
                    "Swim Across": "You tried to swim but the current was too strong.",
                    "Build Raft": "You built a raft and sailed to a small island."
                },
                "next": {
                    "Swim Across": {"terminal_type": "bad"},  # Terminal state: Bad
                    "Build Raft": {
                        "options": ["Explore Island", "Stay on Raft"],
                        "responses": {
                            "Explore Island": "You found a hidden cave with treasures!",
                            "Stay on Raft": "You stayed on the raft and safely returned home."
                        },
                        "next": {
                            "Explore Island": {"terminal_type": "good"},  # Terminal state: Good
                            "Stay on Raft": {"terminal_type": "neutral"}  # Terminal state: Neutral
                        }
                    }
                }
            }
        }
    }
}

class DebuggingGameEnv:
    def __init__(
        self, 
        max_conversation_length: int = 10,
        scenarios: Optional[Dict] = None,
    ):
        self.scenarios = scenarios if scenarios else DEFAULT_GAME_SCENARIOS
        self.scenario_keys = list(self.scenarios.keys())
        self.max_conversation_length = max_conversation_length
        self.random = random.Random(None)
        self.reset()

    def _select_scenario(self):
        scenario_key = self.random.choice(self.scenario_keys)
        scenario = self.scenarios[scenario_key]
        self.current_scenario = scenario
        self.history = INITIAL_STR
        self.turn_count = 0
        self.done = False
        self.current_node = self.current_scenario
        self.history += f"Description: {self.current_node['description']}\n"

        # **Add initial options to the history**
        options_str = ", ".join(self.current_node["options"])
        self.history += f"Options: {options_str}\n"
        return self.history

    def _step(self, choice: str, response: str) -> Tuple[str, float, bool]:
        if self.done:
            return self.history, 0.0, self.done

        self.turn_count += 1
        self.history += f"Choice: {choice}\nOutcome: {response}\n"

        # Navigate to the next node based on the choice
        next_node_info = self.current_node["next"].get(choice)
        if not next_node_info:
            # Invalid choice, end the game
            self.done = True
            reward = -10  # Assign a negative reward for invalid choices
            self.history += "Invalid choice. Game over.\n"
            return self.history, reward, self.done

        if "terminal_type" in next_node_info:
            terminal_type = next_node_info["terminal_type"]
            if terminal_type == "good":
                reward = 10  # Positive reward for winning the game
            elif terminal_type == "bad":
                reward = -10  # Negative reward for bad outcomes
            else:
                reward = 0  # Neutral reward for neutral outcomes
            self.done = True
        else:
            # Continue the game
            self.current_node = next_node_info
            reward = -1  # Negative reward for each step to encourage shorter paths
            self.done = False

        # **Add next options to the history**
            options_str = ", ".join(self.current_node["options"])
            self.history += f"Options: {options_str}\n"

        if self.turn_count >= self.max_conversation_length:
            self.done = True
            reward = 0  # Neutral reward if max length is reached without terminal state

        return self.history, reward, self.done

    def reset(self, idx: Optional[int] = None) -> str:
        return self._select_scenario()

    def copy(self):
        return DebuggingGameEnv(
            max_conversation_length=self.max_conversation_length,
            scenarios=self.scenarios,
        )

class BatchedDebuggingGameEnv:
    def __init__(
        self, 
        env_load_path: str,
        cache_dir: str,
        device: torch.device,
        max_conversation_length: int = 10,
        bsize: int = 4,
        scenarios: Optional[Dict] = None,
    ):
        self.env_list = [DebuggingGameEnv(max_conversation_length, scenarios) for _ in range(bsize)]
        self.bsize = bsize
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
        #self.model.load_state_dict(torch.load(env_load_path)['model_state_dict'])
        self.model.eval()  # Set model to evaluation mode

    def generate_answers(self, prompts: List[str]) -> List[str]:
        inputs = prompts
        encoder_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=encoder_inputs['input_ids'], 
                attention_mask=encoder_inputs['attention_mask'],
                max_new_tokens=50, 
                do_sample=False
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def reset(self, idx: Optional[int] = None) -> List[str]:
        return [env.reset(idx) for env in self.env_list]

    def step(self, choices: List[str]) -> List[Tuple[str, float, bool]]:
        prompts = [
            PROMPT_TEMPLATE.format(history=env.history, options=", ".join(env.current_node["options"]))
            for env in self.env_list
        ]
        answers = self.generate_answers(prompts)
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [
                executor.submit(env._step, choice, answer)
                for env, choice, answer in zip(self.env_list, choices, answers)
            ]
            results = [job.result() for job in jobs]
        return results

def main():
    # Initialize the environment
    env = DebuggingGameEnv()

    # Reset the environment to start a new game
    initial_history = env.reset()
    print(initial_history)

    # Round 1
    choice = "Go to Forest"
    response = "You have entered the Forest."
    history, reward, done = env._step(choice, response)
    print(f"History:\n{history}\nReward: {reward}, Done: {done}\n")

    # Round 2
    choice = "Follow Path"
    response = "You followed the path and encountered a wild animal."
    history, reward, done = env._step(choice, response)
    print(f"History:\n{history}\nReward: {reward}, Done: {done}\n")

    # Round 3 (Terminal State: Bad)
    choice = "Run Away"
    response = "You fought bravely but lost the battle."
    history, reward, done = env._step(choice, response)
    print(f"History:\n{history}\nReward: {reward}, Done: {done}\n")


if __name__ == "__main__":
    main()
