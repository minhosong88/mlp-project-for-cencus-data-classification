from scripts.load_and_preprocess import load_data, preprocess_data, balance_and_split, plot_distribution
from scripts.models import (
    TwoLayerPerceptron,
    ThreeLayerPerceptron,
    FourLayerPerceptron,
    FiveLayerPerceptron,
    FLPRMSProp,
    FLPAdaM
)
from scripts.visualization import plot_gradients, plot_accuracy


def main(model_name="TwoLayerPerceptron"):
    # Load and preprocess data
    df = load_data('data/acs2017_census_tract_data.csv')
    df = balance_and_split(df)
    plot_distribution(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Select model
    if model_name == "TwoLayerPerceptron":
        model = TwoLayerPerceptron(n_hidden=30, epochs=20)
    elif model_name == "ThreeLayerPerceptron":
        model = ThreeLayerPerceptron(n_hidden=30, epochs=20)
    elif model_name == "FourLayerPerceptron":
        model = FourLayerPerceptron(n_hidden=30, epochs=20)
    elif model_name == "FiveLayerPerceptron":
        model = FiveLayerPerceptron(n_hidden=30, epochs=20)
    elif model_name == "FLPRMSProp":
        model = FLPRMSProp(n_hidden=30, epochs=20)
    elif model_name == "FLPAdaM":
        model = FLPAdaM(n_hidden=30, epochs=20)
    else:
        raise ValueError(f"Model {model_name} is not recognized.")

    # Train the selected model
    model.fit(X_train, y_train, print_progress=True, XY_test=(X_test, y_test))

    # Plot results
    plot_accuracy(model.score_, model.val_score_)
    plot_gradients(model.grad_magnitudes)

    # Print final accuracy
    print(f"Final Training Accuracy for {model_name}:", model.score_[-1])
    print(f"Final Validation Accuracy for {model_name}:", model.val_score_[-1])


if __name__ == "__main__":
    # You can change the model name here to test different models
    main(model_name="TwoLayerPerceptron")
