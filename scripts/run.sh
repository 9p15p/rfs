# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
python train_supervised.py --trial pretrain --model_path result/ckpt --tb_path result/tb --data_root dataset/ --batch_size 200

# distillation
# setting '-a 1.0' should give simimlar performance
python train_distillation.py -r 0.5 -a 0.5 --path_t result/ckpt/teacher.pth --trial born1 --model_path result/ckpt --tb_path result/tb --data_root dataset/

# evaluation
python eval_fewshot.py --model_path result/ckpt/student.pth --data_root dataset/
