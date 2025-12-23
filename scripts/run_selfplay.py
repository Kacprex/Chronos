"""Generate self-play games and write RL shards to the replay buffer."""
from src.selfplay.self_play_worker import self_play

if __name__ == "__main__":
    # Defaults mirror hub.py's typical settings; adjust as needed.
    self_play(
        num_games=50,
        simulations=200,
        out_dir=None,          # uses RL_BUFFER_DIR from config
        shard_size=4096,
        temperature_moves=10,
        temperature=1.0,
        initial_temperature=1.25,
        max_moves=512,
        workers=1,
        mcts_batch_size=8,
        infer_max_batch=64,
        infer_wait_ms=3,
    )
