MISTRAL_TWENTY_QUESTIONS_TEMPLATE = """<s>[INST]Please play Twenty Questions with me. The possible hidden words are:
football, dog, banana, truck, pants, computer, piano, chair, pen, scissors.
Some examples are following:
Questions:
Is the object alive? Yes.
Is the object a mammal? No.
Is the object a plant? Yes.
Is the object edible? Yes.
Is the object a fruit? Yes.
Is the object a tropical fruit? Yes.
Is the object a banana? Yes.
You guessed the correct word! You win!

Please continue this conversation by completing the next question. 
{obs}
Please answer in the following format:
{
"Question": "Your Question",
}
The possible hidden words are:
football, dog, banana, truck, pants, computer, piano, chair, pen, scissors.[/INST]
"""

MISTRAL_DEBUG_TEMPLATE = """<s>[INST]You are going to play a text based adventure game.
At each round you will be given a scenario with 2 options to pick from.
You need to pick one of the 2 options and then the game will advance and present you with a new scenario.
Try to chose the correct options to reach the treasure at the end.

An example state would look something like this:

Game History:
Description: You are at the start of the game.
Options: Go to Forest, Go to Lake

You could then answer 'Go to Forest' and the game state would be updated as:

Game History:
Description: You are at the start of the game.
Options: Go to Forest, Go to Lake
Choice: Go to Forest
Outcome: You have entered the Forest.
Options: Eat banana, Go left

Simply chose one of the 2 available options until the game informs you it is finished.

When you respond please answer with only the available options. 
For instance when presented with `Options: Go to Forest, Go to Lake`, answer with either `Go to Forest` or `Go to Lake` and only those words.


Good luck and keep track of the game history here:
{obs}


Only generate your choice. The game will create the rest of the context.[/INST]
"""

#TODO: WHY???
def mistral_debug_decode_actions(output):
    """
    Decode the actions from the output of the model.
    """
    #actions = []
    #for a in output:
    #    action = a.split('"Question":')[-1]
    #    action = action.split("?")[0] + "?"
    #    action = action.strip().replace('"', '')
    #    actions.append(action)
    actions = output
    return actions

#TODO: WHY???
def mistral_twenty_questions_decode_actions(output):
    """
    Decode the actions from the output of the model.
    """
    actions = []
    for a in output:
        action = a.split('"Question":')[-1]
        action = action.split("?")[0] + "?"
        action = action.strip().replace('"', '')
        actions.append(action)
    return actions