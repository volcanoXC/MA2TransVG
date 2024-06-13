import argparse

def load_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--output_dir', type=str,
                        default='output/logs')
    parser.add_argument('--scan_dir', type=str,
                        default='datasets/referit3d/scan_data')
    parser.add_argument('--anno_file', type=str,
                        default='datasets/referit3d/annotations/bert_tokenized/scanrefer.jsonl')
    parser.add_argument('--category_file', type=str,
                        default='datasets/referit3d/annotations/meta_data/scannetv2_raw_categories.json')
    parser.add_argument('--cat2vec_file', type=str,
                        default='datasets/referit3d/annotations/meta_data/cat2glove42b.json')
    parser.add_argument('--train_scan_split', type=str,
                        default='datasets/referit3d/annotations/splits/scannetv2_train.txt')
    parser.add_argument('--val_scan_split', type=str,
                        default='datasets/referit3d/annotations/splits/scannetv2_val.txt')
    parser.add_argument('--tst_scan_split', type=str,
                        default='datasets/referit3d/annotations/splits/scannetv2_test.txt')
    parser.add_argument('--max_txt_len', type=int, default=50)
    parser.add_argument('--max_obj_len', type=int, default=80)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--gpu', type=str, default='0,1,2,3',
                        help='specify gpu device. [default: 0]')
    parser.add_argument('--lr_decay', type=str, default='cosine')
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--lr_decay', type=str, default='cosine')
    parser.add_argument('--grad_norm', type=float, default=5.0)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--txt_encoder_lr_multi', type=float, default=0.1)
    parser.add_argument('--obj_encoder_lr_multi', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--num_obj_classes', type=int, default=607)
    parser.add_argument("--theta", default=0.2, type=float, help="The threshold for cross-replacing attention.")
    parser.add_argument("--cross_dropout", default=0.1, type=float, help="dropout for cross-transformer.")
    parser.add_argument("--skip_connection", action="store_true",
                        help="Whether to add skip connection on cross-transformer")
    parser.add_argument("--use_quantile", action="store_true",
                        help="Whether to use percentage replacing on cross-transformer")
    parser.add_argument('--num_exlayers', type=int, default=6, help='the layers of cross-transformer')
    parser.add_argument('--replace_start', type=int, default=0, help='the start layer of replace in cross-transformer')
    parser.add_argument('--replace_end', type=int, default=5, help='the end layer of replace in cross-transformer')
    parser.add_argument('--num_attlayers', type=int, default=3)
    parser.add_argument('--obj3d_clf_loss', type=int, default=1)
    parser.add_argument('--obj3d_clf_pre_loss', type=int, default=1)
    parser.add_argument('--obj3d_reg_loss', type=int, default=0)
    parser.add_argument('--txt_clf_loss', type=int, default=1)



    args = parser.parse_args()
    return args


