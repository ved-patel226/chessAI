from chess import Board
from termcolor import cprint
import numpy as np
import chess

class Chess_Easy():
    def __init__(self, board: Board = None) -> None:
        if board:
            self.board = board
        else:
            self.board = Board()
            
        cprint(f"Chess_Easy object created", "blue", attrs=["bold"])
    
    def push(self, move: str) -> None:
        self.board.push(move)
    
    def print_board(self) -> None:
        print(self.board)
    
    def board_to_matrix(self, board: Board = None) -> np.ndarray:
        if board is None:
            board = self.board
        
        matrix = np.zeros((13, 8, 8))
        piece_map = board.piece_map()

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1
            piece_color = 0 if piece.color else 6
            matrix[piece_type + piece_color, row, col] = 1

        legal_moves = board.legal_moves
        for move in legal_moves:
            to_square = move.to_square
            row_to, col_to = divmod(to_square, 8)
            matrix[12, row_to, col_to] = 1

        return matrix
  
    def create_input_for_nn(self, games):
        X = []
        y = []
        for game in games:
            board = Board()
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                X.append(self.board_to_matrix(board))
                y.append(move.uci())
                board.push(move)
        return np.array(X, dtype=np.float32), np.array(y)  

    def encode_moves(self, moves):
        move_to_int = {move: idx for idx, move in enumerate(set(moves))}
        return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int


def main() -> None:
    chess_game = Chess_Easy()
    
    board_with_kings = chess_game.board_to_matrix()
    
    print(board_with_kings)
    

if __name__ == '__main__':
    main()
