import flwr as fl
import numpy as np

# Config sent to clients for each round
def fit_config(rnd: int):
    return {
        "round": rnd,
        "batch_size": 32,
        "local_epochs": 1,
    }

# Track accuracy across rounds
accuracy_history = []

# Custom metric aggregator for evaluation
def evaluate_metrics_aggregation(metrics):
    num_examples = [num for num, _ in metrics]
    accuracies = [m["accuracy"] for _, m in metrics]
    weighted_avg_accuracy = np.average(accuracies, weights=num_examples)

    accuracy_history.append(weighted_avg_accuracy)
    print(f"\n📊 Round {len(accuracy_history)} - Average Accuracy: {weighted_avg_accuracy:.4f}")
    return {"accuracy": weighted_avg_accuracy}

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Save accuracy to file
    with open("accuracy_log.txt", "w") as f:
        for i, acc in enumerate(accuracy_history, 1):
            f.write(f"Round {i}: Accuracy = {acc:.4f}\n")
    print("✅ Accuracy log saved to accuracy_log.txt")

if __name__ == "__main__":
    main()
