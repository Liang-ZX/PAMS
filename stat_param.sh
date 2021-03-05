#python statistics.py --model EDSR --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --test_only --cpu > model_param/edsr_x2.log
#python statistics.py --model EDSR --scale 3 --n_resblocks 16 --n_feats 64 --test_only --cpu > model_param/edsr_baseline_x3.log

python statistics.py --model RDN --scale 2 --RDNconfig A --G0 32 --test_only --cpu > model_param/rdn_x2_configA.log
python statistics.py --scale 2 --model RDN --batch_size 16 --patch_size 64 --test_only --gpu_id 2 > ../model_param/rdn_x2.log

