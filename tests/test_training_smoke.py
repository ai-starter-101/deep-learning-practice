from dl_practice.train import train_linear_regression


def test_linear_regression_learns_reasonable_parameters():
    metrics = train_linear_regression(epochs=30, lr=0.05)
    assert abs(metrics["weight"] - 2.0) < 0.5
    assert abs(metrics["bias"] - 1.0) < 0.5
    assert metrics["val_loss"] < 0.2
