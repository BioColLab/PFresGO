import json
import argparse
import numpy as np
from pfresgo.PFresGO import PFresGO
from pfresgo.PFresGO_util import load_GO_annot



if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-hsize', '--hidden_size', type=int, default=128, nargs='+', help="Hidden dimension of decoder layers.")
    parser.add_argument('-hlayer', '--num_hidden_layers', type=int, default=2, nargs='+', help="Number of hidden layers.")
    parser.add_argument('-numhead', '--num_heads', type=int, default=8, nargs='+', help="Number of attention heads.")
    parser.add_argument('-drop', '--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('-lr', type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument('-dff', type=int, default=1024, help="Dimension of ffn layer of decoder.")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs to train.")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help="Batch size.")#32
    parser.add_argument('-pd', '--pad_len', type=int, default=1000, help="Padd length (max len of protein sequences in train set).")
    parser.add_argument('-ont', '--ontology', type=str, default='mf', choices=['mf', 'bp', 'cc'], help="Ontology.")
    parser.add_argument('--model_name', type=str, default='MF_PFresGO', help="Name of the model.")
    parser.add_argument('--autoencoder_name', type=str, default='./trained_model/Autoencoder_128.h5', help="Name of trained autoencoder model.")
    parser.add_argument('--train_tfrecord_fn', type=str,  default='./Datasets/TFRecords_sequences/PDB_GO_train', help='Train tfrecords.')
    parser.add_argument('--valid_tfrecord_fn', type=str, default='./Datasets/TFRecords_sequences/PDB_GO_valid',help='valid tfrecords.')
    parser.add_argument('--annot_fn', type=str, default='./Datasets/nrPDB-GO_2019.06.18_annot.tsv', help="File (*tsv) with GO term annotations.")
    parser.add_argument('--test_list', type=str, default='./Datasets/nrPDB-GO_2019.06.18_test.txt', help="File with test PDB chains.")
    parser.add_argument('--label_embedding_dict', type=str, default='./Datasets/label-embedding-128.npy', help="File with go term embedding.")

    args = parser.parse_args()

    train_tfrecord_fn = args.train_tfrecord_fn + '*'
    valid_tfrecord_fn = args.valid_tfrecord_fn + '*'

    # load annotations
    prot2annot, goterms, gonames, counts = load_GO_annot(args.annot_fn)
    goterms = goterms[args.ontology]
    gonames = gonames[args.ontology]
    output_dim = len(goterms)

    go_emb = []
    noexist_go_emb = []
    go_id_dict = np.load(args.label_embedding_dict, allow_pickle=True)
    es = go_id_dict.item()
    for i in range(len(goterms)):
        if es.get(goterms[i]) is not None:
            go_embedding = es[goterms[i]]
            go_emb.append(go_embedding)
        else:
            noexist_go_emb.append(goterms[i])
    #print("the non existing go embedding is",noexist_go_emb)


    print("### Training model: ", args.model_name, " on ", output_dim, " GO terms.")
    model = PFresGO(output_dim=output_dim, n_channels=26, lr=args.lr, drop=args.dropout,num_hidden_layers=args.num_hidden_layers, train=True, dff = args.dff, batch_size = args.batch_size,
                    model_name_prefix=args.model_name, label_embedding=go_emb, hidden_size=args.hidden_size,num_heads=args.num_heads, autoencoder_name=args.autoencoder_name )

    model.train(train_tfrecord_fn, valid_tfrecord_fn, epochs=args.epochs, batch_size=args.batch_size,pad_len=args.pad_len, ont=args.ontology)

    # save models
    model.save_model('./trained_model/'+args.model_name + '.h5' )
    model.plot_losses()
    print("Finish training")

    # save model params to json
    with open(args.model_name + "_model_params.json", 'w') as fw:
        out_params = vars(args)
        out_params['goterms'] = goterms
        json.dump(out_params, fw, indent=1)












