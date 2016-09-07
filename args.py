import argparse

parser = argparse.ArgumentParser(description="Run GridLSTM or FC DNN on amino acid JSON data.")
parser.add_argument('--data-file', default="data/json_big.txt", type=argparse.FileType('r'),
                    help="JSON file with all the data. Data will be randomly split 90/10 into train/test data.")
parser.add_argument('--hidden-layer-type', choices=["FC", "1GridLSTM", "2GridLSTM"], default="2GridLSTM")
parser.add_argument('--window-size', default=15, type=int,
                    help="Number of acids to look at, including the acid in question in the middle.")
parser.add_argument('--acid-parameter-size', default=21, type=int,
                    help="Number of parameters associated with one acid, including 'dummy' property.")
parser.add_argument('--batch-size', default=128, type=int,
                    help="Number of data points to train in parallel.")
parser.add_argument('--hidden-layer-size', default=50, type=int,
                    help="Number of LSTM units.")
parser.add_argument('--hidden-layer-count', default=5, type=int,
                    help="Number of hidden layers stacked sequentially.")
parser.add_argument('--learning-rate', default=1e-3, type=float,
                    help="Learning rate during Adam training.")
parser.add_argument('--iterations', default=500, type=int,
                    help="Number of training iterations.")
parser.add_argument('--positive-multiplier', metavar="N", default=1, type=int,
                    help="Copy positive examples (unordered) N times in training data.")
parser.add_argument('--positive-error-weight', metavar="N", default=10, type=float,
                    help="During training, exaggerate false-negative (saying ordered when actually unordered) loss by a factor of N.")
parser.add_argument('--test-every', default=20, type=int, metavar="N",
                    help="Measure test performance every Nth iteration.")
parser.add_argument('--save-path-basename', type=str, default="/tmp/model.ckpt", help="Backup model to path every time a test is run (See --test-every).")
parser.add_argument('--load-path', type=str, help="Backup model to path at the end.")
args = parser.parse_args()