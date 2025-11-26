
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.insert(0, "/data/shuaizhang/MobileI2V")

import datetime
import getpass
import hashlib
import json
import os
import os.path as osp
import random
import time
import types
import warnings
from dataclasses import asdict
from pathlib import Path
import pandas as pd

import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from PIL import Image
from termcolor import colored

warnings.filterwarnings("ignore")  # ignore warning


from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import clip_grad_norm_, flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, read_config, set_random_seed
from einops import rearrange
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer
from torchvision.io import write_video
from torchvision import transforms

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_sample(x, save_path=None, fps=8, normalize=True, value_range=(-1, 1), force_video=False, verbose=True):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4
    save_path += ".mp4"
    if normalize:
        low, high = value_range
        x.clamp_(min=low, max=high)
        x.sub_(low).div_(max(high - low, 1e-5))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
    write_video(save_path, x, fps=fps, video_codec="h264")
    if verbose:
        print(f"Saved to {save_path}")
    return save_path

def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaBlock"


@torch.inference_mode()
def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None):
    torch.cuda.empty_cache()
    vis_sampler = config.scheduler.vis_sampler
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[image_height, image_width]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device)

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []

    def run_sampling(init_z=None, label_suffix="", vae=None, sampler="dpm-solver"):
        latents = []
        current_image_logs = []
        if vae is None:
            vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
        for prompt in validation_prompts:
            print(prompt)
            if prompt == "The video features a man in a tuxedo, smiling and looking to the side. The man is well-dressed, wearing a black suit with a white shirt and a black bow tie. He has short, light-colored hair and appears to be in a good mood. The background is blurred, but it seems to be an indoor setting, possibly a room with a wooden floor and walls. The lighting is soft and warm, suggesting an indoor environment. The man's expression and attire suggest a formal or celebratory occasion.":
                ref_image = Image.open("/data/shuaizhang/MobileI2V/test_image.png").convert("RGB")
                flow_score = torch.FloatTensor([1.0]).to(accelerator.device)
            elif prompt == "This person is talking1.":
                ref_image = Image.open("/data/shuaizhang/MobileI2V/test_image.png").convert("RGB")
                flow_score = torch.FloatTensor([1.0]).to(accelerator.device)
            elif prompt == "This person is talking2.":
                ref_image = Image.open("/data/shuaizhang/MobileI2V/test_image.png").convert("RGB")
                flow_score = torch.FloatTensor([2.0]).to(accelerator.device)
            elif prompt == "This person is talking5.":
                ref_image = Image.open("/data/shuaizhang/MobileI2V/test_image.png").convert("RGB")
                flow_score = torch.FloatTensor([5.0]).to(accelerator.device)
            elif prompt == "This person is talking10.":
                ref_image = Image.open("/data/shuaizhang/MobileI2V/test_image.png").convert("RGB")
                flow_score = torch.FloatTensor([10.0]).to(accelerator.device)
            else:
                ref_image = Image.open("/data/shuaizhang/MobileI2V/test_image.png").convert("RGB")
                flow_score = torch.FloatTensor([5.0]).to(accelerator.device)
            transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Resize([720,1280]),
                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),

            ])
            ref_image = transform(ref_image).to(device)
            ref_image = ref_image.unsqueeze(1)
            ref_image = ref_image.unsqueeze(0).half()
            #ref_image = ref_image.unsqueeze(0)


            ref_image = vae_encode(config.vae.vae_type, vae,ref_image,config.vae.sample_posterior,device=device)

            z = (
                torch.randn(1,  config.vae.vae_latent_dim, config.data.num_frames // 8 + 1,latent_height, latent_width, device=device)
                if init_z is None
                else init_z
            )

            ref_image = ref_image.float()
            z[:, :, :1, :, :] = torch.lerp(z[:, :, :1, :, :], ref_image, 1.0)
            '''
            print("ref_image.shape", ref_image.shape)
            conditioning_latent_frames_mask = torch.zeros(
                (config.train.train_batch_size, num_latent_frames),
                dtype=torch.float32,
                device=init_latents.device,
            )
            '''


            embed = torch.load(
                osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
                map_location="cpu",
            )
            caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
            print("caption_embs",caption_embs.shape)
            # caption_embs = caption_embs[:, None]
            # emb_masks = emb_masks[:, None]
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

            if sampler == "dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=4.5,
                    model_kwargs=model_kwargs,
                )
                denoised = dpm_solver.sample(
                    z,
                    steps=config.data.num_frames,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif sampler == "flow_euler":
                flow_solver = FlowEuler(
                    model, condition=caption_embs, uncondition=null_y, cfg_scale=4.5, model_kwargs=model_kwargs
                )
                denoised = flow_solver.sample(z, guide_image=ref_image, flow_score=flow_score, steps=28)
            elif sampler == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=4.5,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                )
                denoised = dpm_solver.sample(
                    z,
                    steps=20,
                    order=2,
                    guide_image=ref_image,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=config.scheduler.flow_shift,
                )
            else:
                raise ValueError(f"{sampler} not implemented")

            latents.append(denoised)
        torch.cuda.empty_cache()
        if vae is None:
            vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
        for prompt, latent in zip(validation_prompts, latents):
            latent = latent.to(torch.float16)
            B, _, D, _, _ = latent.shape
            #latent = rearrange(latent, "B C D H W -> (B D) C H W")
            samples = vae_decode(config.vae.vae_type, vae, latent)
            #samples = rearrange(samples, "(B D) C H W -> B C D H W", B=B, D=D)

            if not os.path.exists(os.path.join(config.work_dir,"vis", str(step))):
                os.makedirs(os.path.join(config.work_dir,"vis", str(step)))
            for i, sample in enumerate(samples):
                save_sample(
                    sample,
                    fps=24,
                    save_path=os.path.join(config.work_dir, "vis", str(step), prompt[:30]),
                    verbose=False
                )

    run_sampling(init_z=None,label_suffix="", vae=vae, sampler=vis_sampler)


    del vae
    flush()
    # return image_logs


def train(config, args, accelerator, model, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0

    # Cache Dataset for BatchSampler
    if args.caching and config.model.multi_scale:
        caching_start = time.time()
        logger.info(
            f"Start caching your dataset for batch_sampler at {cache_file}. \n"
            f"This may take a lot of time...No training will launch"
        )
        train_dataloader.batch_sampler.sampler.set_start(max(train_dataloader.batch_sampler.exist_ids, 0))
        accelerator.wait_for_everyone()
        for index, _ in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            if index % 2000 == 0:
                logger.info(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                print(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
            if (time.time() - caching_start) / 3600 > 3.7:
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
                accelerator.wait_for_everyone()
                break
            if len(train_dataloader.batch_sampler.cached_idx) == len(train_dataloader) - 1000:
                logger.info(
                    f"Saving rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
            accelerator.wait_for_everyone()
            continue
        accelerator.wait_for_everyone()
        print(f"Saving rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
        json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        return

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader.sampler
        )
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # image, json_info, key = batch
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()
            if load_vae_feat:
                z = batch["video"].to(accelerator.device)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(
                        "cuda",
                        enabled=(config.model.mixed_precision == "no" or config.model.mixed_precision == "fp16" or config.model.mixed_precision == "bf16"),
                    ):
                        if config.data.type == "VideoTextDataset":
                            x = batch["video"]
                        else:
                            x = batch[0]
                            x=torch.unsqueeze(x,2)
                        B, _, D, _, _ = x.shape
                        #x = rearrange(x, "B C D H W -> (B D) C H W")
                        #print(x.shape)
                        img_f = x[:,:,0,:,:].unsqueeze(2)
                        #img_f = img_f.repeat(1, 1, 17, 1, 1)
                        #print(img_f.shape)
                        z = vae_encode(
                            config.vae.vae_type, vae, x, config.vae.sample_posterior, accelerator.device
                        )

                        z_f = vae_encode(
                            config.vae.vae_type, vae, img_f, config.vae.sample_posterior, accelerator.device
                        )


            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start

            clean_videos = z

            lm_time_start = time.time()
            if load_text_feat:
                y = batch[1]  # bs, 1, N, C
                y_mask = batch[2]  # bs, 1, 1, N
            else:
                if "T5" in config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        txt_tokens = tokenizer(
                            batch["text"], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).to(accelerator.device)
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                        y_mask = txt_tokens.attention_mask[:, None, None]
                elif (
                    "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    with torch.no_grad():
                        if not config.text_encoder.chi_prompt:
                            max_length_all = config.text_encoder.model_max_length
                            if config.data.type == "VideoTextDataset":
                                prompt = batch["text"]
                            else:
                                prompt = batch[1]
                        else:
                            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                            if config.data.type == "VideoTextDataset":
                                prompt = [chi_prompt + i for i in batch["text"]]
                            else:
                                prompt = [chi_prompt + i for i in batch[1]]
                            #prompt = [chi_prompt + i for i in batch["text"]]
                            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                            max_length_all = (
                                num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
                            )  # magic number 2: [bos], [_]
                        txt_tokens = tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=max_length_all,
                            truncation=True,
                            return_tensors="pt",
                        ).to(accelerator.device)
                        select_index = [0] + list(
                            range(-config.text_encoder.model_max_length + 1, 0)
                        )  # first bos and end N-1
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
                            :, :, select_index
                        ]
                        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
                else:
                    print("error")
                    exit()

            # Sample a random timestep for each image
            bs = clean_videos.shape[0]
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=clean_videos.device
            ).long()
            #timesteps = torch.tensor(900)
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(clean_videos.device)
            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start

            flow_score = batch["flow_score"].to(accelerator.device)

            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                #timesteps = torch.tensor([700]).to("cuda")
                # print("timesteps",timesteps)
                loss_term = train_diffusion.training_losses(
                    model, clean_videos, timesteps, z_f, flow_score, model_kwargs=dict(y=y, mask=y_mask)
                )
                loss = loss_term["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step - sampler.step_start // config.train.train_batch_size
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step
                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, vae:{t_vae:.3f}, lr:{lr:.3e}, "
                )
                # info += (
                #     f"s:({model.module.h}, {model.module.w}), "
                #     if hasattr(model, "module")
                #     else f"s:({model.h}, {model.w}), "
                # )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                lm_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.training_hours
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        optimizer=None,
                        lr_scheduler=None,
                        generator=generator,
                        add_symlink=True,
                    )
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                if (time.time() - training_start_time) / 3600 > config.train.training_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return
 #if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
            if config.train.visualize and (global_step % 10 == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if validation_noise is not None:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                            init_noise=validation_noise,
                        )
                    else:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                        )

            # avoid dead-lock of multiscale data batch sampler
            # for internal, refactor dataloader logic to remove the ad-hoc implementation
            if (
                config.model.multi_scale
                and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
            ):
                global_step = epoch * train_dataloader_len
                logger.info("Early stop current iteration")
                break
            
            '''
            print(batch["path"][0])
            print(float(loss))
            savedata = {
                "path": [batch["path"][0]],
                "loss": [float(loss)]
            }
            df = pd.DataFrame(savedata)

            # 追加写入，不覆盖原有内容
            df.to_csv('loss_output6.csv', mode='a', header=False, index=False, encoding='utf-8')
            '''
            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=global_step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    generator=generator,
                    add_symlink=True,
                )

                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)
        accelerator.wait_for_everyone()


@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, validation_noise, text_encoder, tokenizer
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path,latent_height,latent_width
    global image_size, cache_file, total_steps, image_height, image_width

    config = cfg
    args = cfg
    # config = read_config(args.config)

    training_start_time = time.time()
    load_from = True
    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True,
        )

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.train.use_fsdp:
        init_train = "FSDP"
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        init_train = "DDP"
        fsdp_plugin = None

    #config.model.mixed_precision = "no"
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb
            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")
    #image_size = config.model.image_size
    
    #latent_size = int(image_size) // config.vae.vae_downsample_rate
    image_height = config.model.image_height
    image_width = config.model.image_width
    if (int(image_height) % config.vae.vae_downsample_rate !=0 and int(image_height) % config.vae.vae_downsample_rate !=0):
        latent_height = int(image_height) // config.vae.vae_downsample_rate + 1
        latent_width = int(image_width) // config.vae.vae_downsample_rate + 1
    if (int(image_height) % config.vae.vae_downsample_rate !=0):
        latent_height = int(image_height) // config.vae.vae_downsample_rate + 1
        latent_width = int(image_width) // config.vae.vae_downsample_rate
    elif(int(image_width) % config.vae.vae_downsample_rate !=0):
        latent_height = int(image_height) // config.vae.vae_downsample_rate
        latent_width = int(image_width) // config.vae.vae_downsample_rate + 1
    else:
        latent_height = int(image_height) // config.vae.vae_downsample_rate
        latent_width = int(image_width) // config.vae.vae_downsample_rate
    #print("image_size:",image_size)
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    max_length = config.text_encoder.model_max_length
    validation_noise = (
        torch.randn(1, config.vae.vae_latent_dim, latent_height, latent_width, device="cpu", generator=generator)
        if getattr(config.train, "deterministic_validation", False)
        else None
    )
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
    text_embed_dim = text_encoder.config.hidden_size

    logger.info(f"vae type: {config.vae.vae_type}")
    if config.text_encoder.chi_prompt:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        logger.info(f"Complex Human Instruct: {chi_prompt}")

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )
    if config.train.visualize and len(config.train.validation_prompts):
        # preparing embeddings for visualization. We put it here for saving GPU memory
        valid_prompt_embed_suffix = f"{max_length}token_{config.text_encoder.text_encoder_name}_{text_embed_dim}.pth"
        validation_prompts = config.train.validation_prompts
        skip = True
        if config.text_encoder.chi_prompt:
            uuid_chi_prompt = hashlib.sha256(chi_prompt.encode()).hexdigest()
        else:
            uuid_chi_prompt = hashlib.sha256(b"").hexdigest()
        config.train.valid_prompt_embed_root = osp.join(config.train.valid_prompt_embed_root, uuid_chi_prompt)
        Path(config.train.valid_prompt_embed_root).mkdir(parents=True, exist_ok=True)

        if config.text_encoder.chi_prompt:
            # Save complex human instruct to a file
            chi_prompt_file = osp.join(config.train.valid_prompt_embed_root, "chi_prompt.txt")
            with open(chi_prompt_file, "w", encoding="utf-8") as f:
                f.write(chi_prompt)

        for prompt in validation_prompts:
            prompt_embed_path = osp.join(
                config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
            )
            if not (osp.exists(prompt_embed_path) and osp.exists(null_embed_path)):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            if (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {config.text_encoder.text_encoder_name} ...")
                tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name)

            for prompt in validation_prompts:
                prompt_embed_path = osp.join(
                    config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
                )
                if "T5" in config.text_encoder.text_encoder_name:
                    txt_tokens = tokenizer(
                        prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                    caption_emb_mask = txt_tokens.attention_mask
                elif (
                    "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    if not config.text_encoder.chi_prompt:
                        max_length_all = config.text_encoder.model_max_length
                    else:
                        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                        prompt = chi_prompt + prompt
                        num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                        max_length_all = (
                            num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
                        )  # magic number 2: [bos], [_]

                    txt_tokens = tokenizer(
                        prompt,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][
                        :, select_index
                    ]
                    caption_emb_mask = txt_tokens.attention_mask[:, select_index]
                else:
                    raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")

                if os.path.exists(prompt_embed_path): ##目录存在，返回为真
                    print( 'dir exists' )
                else:
                    print( 'dir not exists')
                    os.makedirs(prompt_embed_path)
                    os.rmdir(prompt_embed_path)
                print(prompt_embed_path)
                torch.save({"caption_embeds": caption_emb, "emb_mask": caption_emb_mask}, prompt_embed_path)

            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            if "T5" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            elif "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            else:
                raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")
            if os.path.exists(null_embed_path): ##目录存在，返回为真
                print( 'dir exists' )
            else:
                print( 'dir not exists')
                os.makedirs(null_embed_path)
                os.rmdir(null_embed_path)
            print(null_embed_path)
            torch.save(
                {"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": null_tokens.attention_mask},
                null_embed_path,
            )
            # if config.data.load_text_feat:
            #     del tokenizer
            #     del text_encoder
            del null_token_emb
            del null_tokens
            flush()

    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # 1. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_v=config.scheduler.predict_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = f"v-prediction: {config.scheduler.predict_v}, noise schedule: {config.scheduler.noise_schedule}"
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 2. build models
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": text_embed_dim,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
    }
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_height=latent_height,
        input_width=latent_width,
        #input_size=latent_size,
        **model_kwargs,
    ).train()
    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )
    # 2-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from

    if config.model.load_from is not None and load_from:
        _, missing, unexpected, _ = load_checkpoint(
            config.model.load_from,
            model,
            load_ema=False,
            null_embed_path=null_embed_path,
        )
        logger.warning(f"Missing keys: {missing}")

        logger.warning(f"Unexpected keys: {unexpected}")
    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # 3. build dataloader
    # config.data.data_dir = config.data.data_dir if isinstance(config.data.data_dir, list) else [config.data.data_dir]
    # config.data.data_dir = [
    #     data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
    #     for data in config.data.data_dir
    # ]
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dataset = build_dataset(
        asdict(config.data),
        # resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        real_prompt_ratio=config.train.real_prompt_ratio,
        max_length=max_length,
        config=config,
        # caption_proportion=config.data.caption_proportion,
        # sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_downsample_rate,
    )
    accelerator.wait_for_everyone()
    if config.model.multi_scale:
        drop_last = True
        uuid = hashlib.sha256("-".join(config.data.data_dir).encode()).hexdigest()[:8]
        cache_dir = osp.expanduser(f"~/.cache/_wids_batchsampler_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_pattern = (
            f"{cache_dir}/{getpass.getuser()}-{uuid}-sort_dataset{config.data.sort_dataset}"
            f"-hq_only{config.data.hq_only}-valid_num{config.data.valid_num}"
            f"-aspect_ratio{len(dataset.aspect_ratio)}-droplast{drop_last}"
            f"dataset_len{len(dataset)}"
        )
        cache_file = f"{base_pattern}-num_replicas{num_replicas}-rank{rank}"
        for i in config.data.data_dir:
            cache_file += f"-{i}"
        cache_file += ".json"

        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.train.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=drop_last,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.data.valid_num,
            hq_only=config.data.hq_only,
            cache_file=cache_file,
            caching=args.caching,
        )
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.train.num_workers)
        train_dataloader_len = len(train_dataloader)
        logger.info(f"rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
    else:
        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        train_dataloader = build_dataloader(
            dataset,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size,
            shuffle=False,
            sampler=sampler,
        )
        train_dataloader_len = len(train_dataloader)
    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # 4. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, rng_state = load_checkpoint(
                **config.model.resume_from,
                model=model,
                optimizer=optimizer if check_flag else None,
                lr_scheduler=lr_scheduler if check_flag else None,
                null_embed_path=null_embed_path,
            )

            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

        # resume randomise
        # if rng_state:
        #     logger.info("resuming randomise")
        #     torch.set_rng_state(rng_state["torch"])
        #     np.random.set_state(rng_state["numpy"])
        #     random.setstate(rng_state["python"])
        #     generator.set_state(rng_state["generator"])  # resume generator status
        #     try:
        #         torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        #     except:
        #         logger.warning("Failed to resume torch_cuda rng state")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
    )


if __name__ == "__main__":

    main()               