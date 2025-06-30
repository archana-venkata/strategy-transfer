
class Move(object):

    def __init__(self, state, action, reward, next_state, done, info):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Move):
            return NotImplemented

        # Move objects are considered equal if they satisfy the following criteria
        # if the rewards are equal
        return self.reward == __o.reward and self.info == __o.info

    def __str__(self) -> str:
        return str(self.info)
        # return 'State:' + str(self.state) + ' Action:' + str(self.action) + ' Reward:' + str(self.reward) + ' Next State:' + str(self.next_state)

    def get_info(self):
        return self.info["result_of_action"]

    def get_char_representation(self):
        if self.info["result_of_action"] == '':
            return ''
        else:
            return self.info["result_of_action"][0].upper()
