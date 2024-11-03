import chess.pgn
from chess import pgn
from chess_easy import Chess_Easy
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np
from termcolor import cprint

def load_pgn(file_path) -> list:
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        cprint("GPU is recognized and available.", "green", attrs=["bold"])
        cprint(f"Available GPUs: {[device.name for device in physical_devices]}", "green", attrs=["bold"])
    else:
        cprint("No GPU found. Using CPU instead.", "yellow", attrs=["bold"])

def main() -> None:
    check_gpu()
    
    chess_game = Chess_Easy()

    files = [file for file in os.listdir("pgn") if file.endswith(".pgn")]
    LIMIT_OF_FILES = min(len(files), 5)
    games = []
    i = 1
    for file in tqdm(files):
        games.extend(load_pgn(f"pgn/{file}"))
        if i >= LIMIT_OF_FILES:
            break
        i += 1

    print(f"GAMES PARSED: {len(games)}")

    X, y = chess_game.create_input_for_nn(games)
    print(f"NUMBER OF SAMPLES: {len(y)}")

    X = X[:2500000]
    y = y[:2500000]

    y, move_to_int = chess_game.encode_moves(y)
    num_classes = len(move_to_int)

    X = np.array(X)
    X = X.reshape(X.shape[0], -1)

    input_shape = (X.shape[1],)

    model = build_model(input_shape, num_classes)
    model.summary()

    model.fit(X, y, epochs=10, batch_size=32)
    
    model.save("chess_model")

if __name__ == '__main__':
    main()
