import numpy as np
import warnings
import h5py
import argparse
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from source_code.AI.datasets import ECGSequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')
    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)

    # Apply threshold to predicted scores
    threshold = 0.4
    y_pred = np.where(y_score > threshold, 1, 0)
    print(y_pred.shape)
    # Load true labels from the file
    import pandas as pd
    df=pd.read_csv('test_annotations.csv')
    print(df.shape)
    # # Calculate accuracy
    accuracy = accuracy_score(df, y_pred)
    print(f"Accuracy on test data: {accuracy*100:.4f}")

    # Generate dataframe
    np.save(args.output_file, y_pred)

    print("Output predictions saved")
