CUDA_VISIBLE_DEVICES=0 python scripts/inference_i2v.py \
      --config=./configs/mobilei2v_config/MobileI2V_300M_img512.yaml \
      --save_path=humface_1126 \
      --model_path=./model/hybrid_371.pth \
      --txt_file=asset/test.txt \
      --flow_score=2.0 \
      --prompt=""