import torch

from app.rl_text_model import RLTextGenerator, build_rl_training_corpus


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    corpus = build_rl_training_corpus()
    generator = RLTextGenerator(corpus, device=device)

    print("Warm-starting the text model with supervised next-token training...")
    generator.train_supervised(
        examples=corpus,
        epochs=250,
        learning_rate=0.01
    )

    print("Post-training with REINFORCE using the formatting reward...")
    reward_history = generator.train_with_reinforce(
        episodes=500,
        max_words=24,
        learning_rate=0.002,
        temperature=0.9
    )

    generator.save("rl_text_model.pth")
    print("Saved rl_text_model.pth")

    test_output = generator.generate_text(
        "How does reinforcement learning help text generation?",
        max_words=24
    )
    print(f"Example output: {test_output}")
    print(f"Final average reward: {sum(reward_history[-50:]) / 50:.3f}")


if __name__ == "__main__":
    main()
