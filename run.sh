DATA_DIR="./dataset"

edsr_x4() {
python main.py --scale 4 \
--k_bits $1 --model EDSR \
--pre_train ./pretrained/edsr_baseline_x4.pt --patch_size 192 \
--data_test Set14 \
--save "output/edsr_x4/$1bit" --dir_data $DATA_DIR --print_every 10 --reset
}

#edsr_x4 8

edsr_x2() {
python main.py --scale 2 \
--k_bits $1 --model EDSR \
--pre_train ./pretrained/edsr_baseline_x2.pt \
--data_test Set14+Set5+B100+Urban100 \
--save "output/edsr_x2/$1bit" --dir_data $DATA_DIR --print_every 100 --reset
}

edsr_x2 8


edsr_x4_eval() {
python3 main.py --scale 4 --model EDSR \
--k_bits $1 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--refine $2 \
--save "output/edsr_x4/$1bit" --dir_data $DATA_DIR 
}

# edsr_x4_eval 8 ./pretrained/8bit_edsr_x4.pt
#
#edsr_x4_save() {
#python3 main.py --scale 4 --model EDSR \
#--k_bits $1 --test_only \
#--data_test Set14 \
#--save "experiment/output/edsr_x4/$1bit" --dir_data $DATA_DIR
#}
#
#edsr_x4_save 8


rdn_x4() {
python main.py --scale 4 \
--k_bits $1 --model RDN \
--pre_train ./pretrained/rdn_baseline_x4.pt  --patch_size 96 \
--data_test Set14 \
--save "output/rdn_x4/$1bit" --dir_data $DATA_DIR
}

# rdn_x4 8

rdn_x4_eval() {
python3 main.py --scale 4 --model RDN \
--k_bits $1  --save_results --test_only \
--data_test Set5+Set14+B100+Urban100 \
--refine $2 \
--save "output/rdn_x4/$1bit" --dir_data $DATA_DIR
}
 
# rdn_x4_eval 4 ./pretrained/4bit_rdn_x4.pt
