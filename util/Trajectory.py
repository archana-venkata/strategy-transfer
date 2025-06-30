from util.Move import Move


class Trajectory(object):
    def __init__(self, moves=[]):
        self.moves: Move = moves.copy()

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Trajectory):
            return NotImplemented
        sequence1 = "".join(self.get_moves_as_chars())
        sequence2 = "".join(__o.get_moves_as_chars())
        return sequence1 == sequence2

    def add(self, move: Move):
        self.moves.append(move)

    def get_moves(self):
        return self.moves

    def get_moves_info(self):
        return [move.get_info() if move != '-' else '-' for move in self.moves]

    def get_moves_as_chars(self):
        return [move.get_char_representation() for move in self.moves if move != '-']
