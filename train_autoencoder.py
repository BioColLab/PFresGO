import argparse
from autoencoder import AutoEncoder


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-idim', '--input_dims', type=int, default=1024, nargs='+', help="Dimensions of input.")
    parser.add_argument('-hdim', '--hid_dims', type=int, default=[1024,256,128], nargs='+', help="Dimensions of middle layers.")
    parser.add_argument('-lr', type=float, default=0.0001, help="learning rate.")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs to train.")
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help="Batch size.")
    parser.add_argument('--model_name', type=str, default='Autoencoder_128', help="Name of the autoencoder model.")
    parser.add_argument('--train_tfrecord_fn', type=str, default='./Datasets/TFRecords_sequences/PDB_GO_train', help='Train tfrecords.')
    parser.add_argument('--valid_tfrecord_fn', type=str, default='./Datasets/TFRecords_sequences/PDB_GO_valid', help='valid tfrecords.')
    args = parser.parse_args()

    train_tfrecord_fn = args.train_tfrecord_fn + '*'
    valid_tfrecord_fn = args.valid_tfrecord_fn + '*'

    model = AutoEncoder(input_dim=args.input_dims, hidden_dims=args.hid_dims, learning_rate=args.lr, batch_size=args.batch_size, num_steps=args.epochs,model_name=args.model_name)
    model.train(train_tfrecord_fn, valid_tfrecord_fn)

    # save models
    model.save_model()
    print("Finished!")






